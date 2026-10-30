package xendroid.compose.core

import java.io.File
import java.util.zip.ZipEntry
import java.util.zip.ZipInputStream
import java.util.zip.ZipOutputStream
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class SanitizedSessionExportTest {
    @get:Rule val folder = TemporaryFolder()

    @Test fun redactsCurrentAndArchivedLogsWithoutEmbeddingOriginalZipOrBinaryTrace() {
        val history = File(folder.root, "session_20260929-100000.zip")
        ZipOutputStream(history.outputStream()).use { zip ->
            zip.putNextEntry(ZipEntry("xe.log"))
            zip.write("Gamertag: Alice Spark\nxuid: 0123456789abcdef\n".toByteArray())
            zip.closeEntry()
            zip.putNextEntry(ZipEntry("exit-trace-0.tombstone.pb"))
            zip.write("BINARY_PRIVATE_DATA".toByteArray())
            zip.closeEntry()
        }
        val live = folder.newFile("xe.log").apply {
            writeText("ip=192.168.1.5 email=a@example.org access_token=secret1234\n" +
                "{\"access_token\":\"jsonSecret123\"} Bearer bearerSecret99\n" +
                "file /storage/emulated/0/ROMs/MyGame.iso\n" + "x".repeat(9000) + "\n")
        }
        val result = SanitizedSessionExport.create(
            File(folder.root, "share.zip"), listOf(history), listOf(live to "xe.log"),
        )!!

        val entries = mutableMapOf<String, String>()
        ZipInputStream(result.inputStream()).use { zip ->
            while (true) {
                val entry = zip.nextEntry ?: break
                entries[entry.name] = zip.readBytes().toString(Charsets.UTF_8)
            }
        }
        assertEquals(setOf("sessions/session_20260929-100000/xe.log", "current/xe.log"), entries.keys)
        val text = entries.values.joinToString("\n")
        for (privateValue in listOf("Alice Spark", "0123456789abcdef", "192.168.1.5",
            "a@example.org", "secret1234", "jsonSecret123", "bearerSecret99",
            "MyGame.iso", "BINARY_PRIVATE_DATA")) {
            assertFalse("Leaked $privateValue", text.contains(privateValue))
        }
        assertTrue(text.contains("[identity]"))
        assertTrue(text.contains("[credential]"))
        assertTrue(text.contains("[storage-path]"))
        assertTrue(text.contains("[long line omitted]"))
        assertTrue(history.isFile && live.readText().contains("secret1234"))
    }

    @Test fun xeniaProfileLogFormatsDoNotLeakXuidOrGamertag() {
        val lines = listOf(
            "i> F8000004 Loaded Alice Spark (GUID: E000000012345678) to slot 0",
            "w> F8000004 User Bob Stone (XUID: E000000087654321) doesn't have profile GPD!",
            "i> F8000004 LoadAccount: Loading Account: E000000011112222",
            "i> F8000004 FindProfiles: Adding profile E000000055556666 to profile list",
            "logged_profile_slot_0_xuid = \"E000000033334444\"",
            "xuid=0xE000000077778888",
        )
        val text = lines.joinToString("\n") { LogRedactor.redact(it) }
        for (secret in listOf("Alice Spark", "Bob Stone", "E000000012345678", "E000000087654321",
            "E000000011112222", "E000000055556666", "E000000033334444", "E000000077778888")) {
            assertFalse("Leaked $secret in:\n$text", text.contains(secret))
        }
        assertTrue(text.contains("logged_profile_slot_0_xuid = [identity]"))
        // Diagnostic hashes without an identity keyword stay readable.
        assertEquals("Translated VS 0123456789ABCDEF in 3 ms",
            LogRedactor.redact("Translated VS 0123456789ABCDEF in 3 ms"))
    }

    @Test fun allowedEntriesAfterABinaryTraceAreStillExported() {
        val history = File(folder.root, "session_20260930-080000.zip")
        ZipOutputStream(history.outputStream()).use { zip ->
            zip.putNextEntry(ZipEntry("exit-trace-0.tombstone.pb")); zip.write(ByteArray(64) { 7 }); zip.closeEntry()
            zip.putNextEntry(ZipEntry(SessionLogs.leftoverEntryName(0, "session_20260929-221500-xe.log")))
            zip.write("leftover log line\n".toByteArray()); zip.closeEntry()
            zip.putNextEntry(ZipEntry("exit-info.txt")); zip.write("reason=CRASH_NATIVE\n".toByteArray()); zip.closeEntry()
        }
        val result = SanitizedSessionExport.create(File(folder.root, "after-binary.zip"), listOf(history), emptyList())!!
        val names = mutableListOf<String>()
        ZipInputStream(result.inputStream()).use { zip -> while (true) names.add((zip.nextEntry ?: break).name) }
        assertEquals(listOf(
            "sessions/session_20260930-080000/previous-0-20260929-221500-xe.log",
            "sessions/session_20260930-080000/exit-info.txt"), names)
    }

    @Test fun leftoverEntryNamesAreUniqueAcrossFailedShelves() {
        val names = listOf("session_20260929-221500-xe.log", "session_20260929-221500-logcat.txt",
            "session_20260930-090000-xe.log", "session_odd-xe.log").mapIndexed(SessionLogs::leftoverEntryName)
        assertEquals(names.size, names.toSet().size)
        assertTrue(names.none { it == "xe.log" || it == "logcat.txt" })
    }

    @Test fun clockTimesStayAndNetworkAddressesGo() {
        // Every shared logcat line used to lose its time to the IPv6 pattern.
        assertEquals("10-27 00:01:55.215 10194 10268 I SessionLogs: Shelved",
            LogRedactor.redact("10-27 00:01:55.215 10194 10268 I SessionLogs: Shelved"))
        assertEquals("ae::surface_detach: marshalling DetachSurface to main_thr",
            LogRedactor.redact("ae::surface_detach: marshalling DetachSurface to main_thr"))
        assertEquals("addr [ip] up", LogRedactor.redact("addr 2001:0db8:85a3:0000:0000:8a2e:0370:7334 up"))
        assertEquals("peer [ip]%wlan0, dns [ip], lo [ip]",
            LogRedactor.redact("peer fe80::1%wlan0, dns 2001:db8::8a2e:370:7334, lo ::1"))
        assertEquals("bt [mac] paired", LogRedactor.redact("bt AA:BB:cc:dd:ee:0F paired"))
        // Followed by a colon, a port or a prefix length, in short form, or after "::" still go.
        assertEquals("BluetoothDevice [mac]: bond state changed",
            LogRedactor.redact("BluetoothDevice AA:BB:CC:DD:EE:FF: bond state changed"))
        assertEquals("neighbor [ip] reachable", LogRedactor.redact("neighbor 2804:14c:5b81:8a1f:1d2e:3c4b:5a69:7f80: reachable"))
        assertEquals("Failed to connect to /[ip]", LogRedactor.redact("Failed to connect to /2804:14c:5b81:8a1f:1d2e:3c4b:5a69:7f80:443"))
        assertEquals("prefix [ip]/64", LogRedactor.redact("prefix 2804:14c:5b81:8a1f/64"))
        assertEquals("ether [ip]", LogRedactor.redact("ether 0:1a:2b:3c:4d:5e"))
        assertEquals("[ip]: reachable", LogRedactor.redact("2001:db8::1: reachable"))
        assertEquals("a64::A64Emitter std::string", LogRedactor.redact("a64::A64Emitter std::string"))
    }

    @Test fun emptyInputDoesNotCreateShareArchive() {
        val dest = File(folder.root, "empty.zip")
        assertEquals(null, SanitizedSessionExport.create(dest, emptyList(), emptyList()))
        assertFalse(dest.exists())
    }

    @Test fun aLongLogKeepsItsStartAndItsEnd() {
        // A run that logs its kernel calls: the GPU, driver and settings lines would fall out of
        // a tail-only export.
        val live = longLog("xe.log",
            "i> f:0000000 00002982 Vulkan device 'Adreno (TM) 750': API 1.4.359 (1.3 used)\n" +
                "i> f:0000000 000036A8   render_target_path = 'accuracy'\n",
            "x> f:0008388 F8000064 Guest crashed at PC 0x82583D1C\n")
        val text = exportCurrent(live, "xe.log")
        assertTrue(text.startsWith("i> f:0000000 00002982 Vulkan device 'Adreno (TM) 750'"))
        assertTrue(text.contains("render_target_path = 'accuracy'"))
        assertEquals(1, Regex("\\[log middle omitted]\n").findAll(text).count())
        assertTrue(text.endsWith("Guest crashed at PC 0x82583D1C\n"))
        assertFalse(text.contains("[diagnostics truncated]"))
        assertTrue(text.length < live.length() - 512 * 1024)
    }

    @Test fun theStartOfALongLogEndsAtAWholeLine() {
        // The line across the start's limit is left out, not cut: a cut keeps part of a value
        // that no pattern recognizes any more, like the start of an e-mail address.
        val first = "i> first line\n"
        val pad = "p".repeat(SanitizedSessionExport.HEAD_SIZE - first.length - 20) + "\n"
        val live = longLog("xe.log", first + pad + "i> contact someone@example.org about it\n", "last line\n")
        val text = exportCurrent(live, "xe.log")
        assertTrue(text.startsWith("i> first line\n[long line omitted]\n[log middle omitted]\n"))
        assertFalse(text.contains("someone@"))
        assertTrue(text.endsWith("last line\n"))
    }

    @Test fun shortLogsAndLongLogcatsKeepTheirOldShape() {
        val short = folder.newFile("xe.log").apply { writeText("i> one\ni> two\n") }
        assertEquals("i> one\ni> two\n", exportCurrent(short, "xe.log"))
        val logcat = longLog("logcat.txt", "10-21 13:43:48.620 first\n", "10-21 13:50:00.000 last\n")
        val text = exportCurrent(logcat, "logcat.txt")
        assertFalse(text.contains("first"))
        assertFalse(text.contains("[log middle omitted]"))
        assertTrue(text.endsWith("10-21 13:50:00.000 last\n"))
    }

    @Test fun aLongLogNeverRepeatsALine() {
        // Three bytes a character and only 896 bytes over the limit: a start counted in
        // characters ended past where the end, counted in bytes, began.
        val live = File(folder.root, "xe.log").apply {
            bufferedWriter().use { out -> repeat(182_400) { out.write("日本語 line %07d\n".format(it)) } }
        }
        val text = exportCurrent(live, "xe.log")
        val numbers = Regex("日本語 line (\\d{7})").findAll(text).map { it.groupValues[1].toInt() }.toList()
        assertEquals(0, numbers.first())
        assertEquals(182_399, numbers.last())
        assertTrue(numbers.zipWithNext().all { (a, b) -> b > a })
        assertEquals(1, Regex("\\[log middle omitted]\n").findAll(text).count())
    }

    @Test fun anArchivedSessionKeepsItsLogsEndAndItsOtherFiles() {
        // A run that crashed is shelved when the app starts again: the end of its log, with the
        // crash, and the files after it in the archive used to be left out.
        val history = File(folder.root, "session_20261028-114106.zip")
        ZipOutputStream(history.outputStream()).use { zip ->
            zip.putNextEntry(ZipEntry("xe.log"))
            zip.write("i> f:0000000 00002982 Vulkan device 'Xclipse 940'\n".toByteArray())
            repeat(110_000) { zip.write("d> f:0001906 F8001780 XamVoiceIsActiveProcess()\n".toByteArray()) }
            zip.write("x> f:0008388 F8000064 Guest crashed at PC 0x82583D1C\n".toByteArray())
            zip.closeEntry()
            zip.putNextEntry(ZipEntry("logcat.txt"))
            zip.write("10-28 11:41:06.107 1 1 I a: b\n".toByteArray())
            zip.closeEntry()
            zip.putNextEntry(ZipEntry("exit-info.txt"))
            zip.write("reason=CRASH_NATIVE\n".toByteArray())
            zip.closeEntry()
        }
        val result = SanitizedSessionExport.create(File(folder.root, "share.zip"), listOf(history), emptyList())!!
        val entries = linkedMapOf<String, String>()
        ZipInputStream(result.inputStream()).use { zip ->
            while (true) {
                val entry = zip.nextEntry ?: break
                entries[entry.name] = zip.readBytes().toString(Charsets.UTF_8)
            }
        }
        val dir = "sessions/session_20261028-114106"
        assertEquals(listOf("$dir/xe.log", "$dir/logcat.txt", "$dir/exit-info.txt"), entries.keys.toList())
        val log = entries.getValue("$dir/xe.log")
        assertTrue(log.startsWith("i> f:0000000 00002982 Vulkan device 'Xclipse 940'\n"))
        assertEquals(1, Regex("\\[log middle omitted]\n").findAll(log).count())
        assertTrue(log.endsWith("Guest crashed at PC 0x82583D1C\n"))
        assertEquals("10-28 11:41:06.107 1 1 I a: b\n", entries.getValue("$dir/logcat.txt"))
        assertEquals("reason=CRASH_NATIVE\n", entries.getValue("$dir/exit-info.txt"))
    }

    /** [start], then about 5 MB of kernel call lines, then [end]. */
    private fun longLog(name: String, start: String, end: String): File = File(folder.root, name).apply {
        bufferedWriter().use { out ->
            out.write(start)
            repeat(110_000) { out.write("d> f:0001906 F8001780 XamVoiceIsActiveProcess()\n") }
            out.write(end)
        }
    }

    private fun exportCurrent(file: File, name: String): String {
        val result = SanitizedSessionExport.create(File(folder.root, "share-$name.zip"), emptyList(), listOf(file to name))!!
        ZipInputStream(result.inputStream()).use { zip ->
            assertEquals("current/$name", zip.nextEntry!!.name)
            return zip.readBytes().toString(Charsets.UTF_8)
        }
    }
}
