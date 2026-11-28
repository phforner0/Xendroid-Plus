package xendroid.compose.archive

import java.io.File
import java.nio.file.Files
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream
import org.junit.Assert.*
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class ArchiveFilesTest {
    @get:Rule val folder = TemporaryFolder()
    private fun archive(vararg entries: Pair<String, ByteArray>): File = folder.newFile().also { f ->
        ZipOutputStream(f.outputStream()).use { zip -> entries.forEach { (name, bytes) ->
            zip.putNextEntry(ZipEntry(name)); zip.write(bytes); zip.closeEntry()
        } }
    }
    @Test fun traversalAndCaseCollisionsAreRejectedBeforeTheyCanEscape() {
        for (entries in listOf(arrayOf("../outside" to byteArrayOf(1)),
            arrayOf("..\\outside" to byteArrayOf(1)),
            arrayOf("/etc/passwd" to byteArrayOf(1)),
            arrayOf("\\etc\\passwd" to byteArrayOf(1)),
            arrayOf("C:\\evil" to byteArrayOf(1)),
            arrayOf("dir/./../outside" to byteArrayOf(1)),
            arrayOf("dir/file" to byteArrayOf(1), "DIR/FILE" to byteArrayOf(2)))) {
            val dest = folder.newFolder()
            assertThrows(IllegalArgumentException::class.java) { ArchiveFiles.unpack(archive(*entries), dest, ArchiveLimits()) }
            assertFalse(File(folder.root, "outside").exists())
        }
    }
    @Test fun backslashSeparatorsAndLeadingDotSlashEntriesUnpackCleanly() {
        val f = folder.newFile().also { file ->
            ZipOutputStream(file.outputStream()).use { zip ->
                zip.putNextEntry(ZipEntry("./")); zip.closeEntry()
                zip.putNextEntry(ZipEntry("sub\\")); zip.closeEntry()
                zip.putNextEntry(ZipEntry("./meta.json")); zip.write("{}".toByteArray()); zip.closeEntry()
                zip.putNextEntry(ZipEntry("sub\\file.bin")); zip.write(byteArrayOf(7, 8)); zip.closeEntry()
            }
        }
        val dest = folder.newFolder()
        val unpacked = ArchiveFiles.unpack(f, dest, ArchiveLimits())
        assertEquals(2, unpacked.size)
        assertEquals("{}", File(dest, "meta.json").readText())
        assertArrayEquals(byteArrayOf(7, 8), File(dest, "sub/file.bin").readBytes())
    }
    @Test fun actualDecodedBytesAndEntryCountAreBounded() {
        val archive = archive("large" to ByteArray(4096))
        assertThrows(IllegalArgumentException::class.java) {
            ArchiveFiles.unpack(archive, folder.newFolder(), ArchiveLimits(entryBytes = 256))
        }
    }
    @Test fun destinationLinksAreNeverFollowed() {
        val dir = folder.newFolder()
        val outside = folder.newFolder()
        Files.createSymbolicLink(File(dir, "link").toPath(), outside.toPath())
        assertThrows(IllegalArgumentException::class.java) { ArchiveFiles.resolve(dir, "link/file") }
        ArchiveFiles.deleteTree(dir)
        assertTrue(outside.isDirectory)
    }
    @Test fun contentLeasePreventsTwoConcurrentOwners() {
        val root = folder.newFolder()
        ContentLease.acquire(root).use {
            assertThrows(ContentBusyException::class.java) { ContentLease.acquire(root) }
        }
        ContentLease.acquire(root).use { }
    }

    /** Tries the lease's lock file from ANOTHER process (fcntl, the same locks the JVM uses on
     *  Linux): "busy", "free", or null when python3 is not there to try. */
    private fun otherProcessTries(lockFile: java.io.File): String? = runCatching {
        val process = ProcessBuilder("python3", "-c",
            "import fcntl,sys\nf=open(sys.argv[1],'r+')\ntry:\n  fcntl.lockf(f, fcntl.LOCK_EX|fcntl.LOCK_NB)\n" +
                "  print('free')\nexcept OSError:\n  print('busy')", lockFile.path)
            .redirectErrorStream(true).start()
        check(process.waitFor(20, java.util.concurrent.TimeUnit.SECONDS))
        process.inputStream.bufferedReader().readText().trim()
    }.getOrNull()?.takeIf { it == "busy" || it == "free" }

    @Test fun aRefusedSecondLeaseInTheSameProcessDoesNotFreeTheFirst() {
        val root = folder.newFolder()
        val lockFile = java.io.File(root, ".content-session.lock")
        val first = ContentLease.acquire(root)
        val whileHeld = try {
            // Each refused attempt used to open and close the lock file, and closing any descriptor
            // drops every POSIX lock of the process: the game process could then take the lease.
            repeat(3) { assertThrows(ContentBusyException::class.java) { ContentLease.acquire(root) } }
            otherProcessTries(lockFile)
        } finally {
            first.close()
        }
        org.junit.Assume.assumeTrue("python3 is needed to try the lock from another process", whileHeld != null)
        assertEquals("busy", whileHeld)
        assertEquals("free", otherProcessTries(lockFile))
        first.close()                                       // closing twice is harmless
        ContentLease.acquire(root).use { }
    }

    @Test fun busyLeaseIsRetriedUntilReleasedWithoutBusyWaiting() = kotlinx.coroutines.test.runTest {
        var clock = 0L
        val sleeps = mutableListOf<Long>()
        var busyNotices = 0
        var attempts = 0
        val result = acquireWithRetry(timeoutMs = 1_000, pollMs = 250, now = { clock },
            sleep = { sleeps.add(it); clock += it }, onBusy = { busyNotices++ }) {
            if (++attempts < 3) throw ContentBusyException() else "lease"
        }
        assertEquals("lease", result)
        assertEquals(listOf(250L, 250L), sleeps)
        assertEquals(1, busyNotices)
    }

    @Test fun busyLeaseGivesUpAtTheDeadlineAndOtherErrorsAreNotRetried() = kotlinx.coroutines.test.runTest {
        var clock = 0L
        var attempts = 0
        assertThrows(ContentBusyException::class.java) {
            kotlinx.coroutines.runBlocking {
                acquireWithRetry(timeoutMs = 600, pollMs = 250, now = { clock }, sleep = { clock += it }) {
                    attempts++; throw ContentBusyException()
                }
            }
        }
        // 0, 250, 500 and the final attempt at the 600 ms deadline.
        assertEquals(4, attempts)
        attempts = 0
        assertThrows(java.io.IOException::class.java) {
            kotlinx.coroutines.runBlocking {
                acquireWithRetry<Unit>(timeoutMs = 600, now = { clock }, sleep = { clock += it }) {
                    attempts++; throw java.io.IOException("disk")
                }
            }
        }
        assertEquals(1, attempts)
    }
}
