package xendroid.compose.driver

import java.io.File
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream
import org.junit.Assert.*
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import xendroid.compose.archive.ArchiveFiles

class DriverPackageInstallerTest {
    @get:Rule val folder = TemporaryFolder()
    private fun elf(marker: Byte): ByteArray = ByteArray(128).apply {
        this[0] = 0x7f; this[1] = 'E'.code.toByte(); this[2] = 'L'.code.toByte(); this[3] = 'F'.code.toByte()
        this[4] = 2; this[5] = 1; this[6] = 1; this[16] = 3; this[18] = 183.toByte(); this[80] = marker
    }
    private fun zip(bytes: ByteArray, libraryName: String = "libvulkan.so", path: String = "libvulkan.so"): File = folder.newFile().also { f ->
        ZipOutputStream(f.outputStream()).use {
            it.putNextEntry(ZipEntry("meta.json")); it.write("{\"libraryName\":\"$libraryName\"}".toByteArray()); it.closeEntry()
            it.putNextEntry(ZipEntry(path)); it.write(bytes); it.closeEntry()
        }
    }
    @Test fun importsAreImmutableAndSameArchiveIsReusedOnlyIfVerified() {
        val installer = DriverPackageInstaller(folder.newFolder())
        val archive = zip(elf(1))
        val first = installer.install(archive, ArchiveFiles.sha256(archive))
        assertTrue(first.verifiedDownload)
        assertEquals(first.library, installer.install(archive).library)
        val second = installer.install(zip(elf(2)))
        assertNotEquals(first.library.parent, second.library.parent)
        assertArrayEquals(elf(1), first.library.readBytes())
    }
    @Test fun wrongAbiDigestAndUnsafeMetadataCannotInstall() {
        for (archive in listOf(zip(elf(1).apply { this[18] = 62 }), zip(elf(1), "../libvulkan.so"))) {
            val root = folder.newFolder()
            assertThrows(IllegalArgumentException::class.java) { DriverPackageInstaller(root).install(archive) }
            assertFalse(root.listFiles()!!.any { it.isDirectory })
        }
        assertThrows(IllegalArgumentException::class.java) { DriverPackageInstaller(folder.newFolder()).install(zip(elf(1)), "0".repeat(64)) }
    }
    @Test fun interruptedExtractionLeavesInstalledDriverUntouched() {
        val root = folder.newFolder()
        val installer = DriverPackageInstaller(root)
        val previous = installer.install(zip(elf(1)))
        var calls = 0
        assertThrows(java.io.IOException::class.java) {
            installer.install(zip(elf(2))) { if (++calls == 3) throw java.io.IOException("cancel") }
        }
        assertArrayEquals(elf(1), previous.library.readBytes())
        assertFalse(root.listFiles()!!.any { it.name.startsWith(".install-") })
    }
    @Test fun nestedFolderWithWindowsPathsAndHelperSosInstallsAndRejectsNestedEscape() {
        val nestedZip = folder.newFile().also { f ->
            ZipOutputStream(f.outputStream()).use { zip ->
                zip.putNextEntry(ZipEntry("./")); zip.closeEntry()
                zip.putNextEntry(ZipEntry("ExynosTools_V2.0\\meta.json"))
                zip.write("{\"libraryName\":\"vulkan.xclipse.so\"}".toByteArray())
                zip.closeEntry()
                zip.putNextEntry(ZipEntry("ExynosTools_V2.0\\vulkan.xclipse.so"))
                zip.write(elf(9))
                zip.closeEntry()
                zip.putNextEntry(ZipEntry("ExynosTools_V2.0\\libhelper.so"))
                zip.write(elf(10))
                zip.closeEntry()
            }
        }
        val installed = DriverPackageInstaller(folder.newFolder()).install(nestedZip)
        assertEquals("vulkan.xclipse.so", installed.library.name)
        assertArrayEquals(elf(9), installed.library.readBytes())

        val nestedEscape = folder.newFile().also { f ->
            ZipOutputStream(f.outputStream()).use { zip ->
                zip.putNextEntry(ZipEntry("sub/meta.json"))
                zip.write("{\"libraryName\":\"../libvulkan.so\"}".toByteArray())
                zip.closeEntry()
                zip.putNextEntry(ZipEntry("libvulkan.so"))
                zip.write(elf(3))
                zip.closeEntry()
            }
        }
        assertThrows(IllegalArgumentException::class.java) {
            DriverPackageInstaller(folder.newFolder()).install(nestedEscape)
        }
    }
}
