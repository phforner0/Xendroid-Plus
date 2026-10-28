package xendroid.compose.core

import java.io.File
import java.io.FileInputStream
import java.io.InputStream
import java.io.InputStreamReader
import java.util.zip.ZipEntry
import java.util.zip.ZipInputStream
import java.util.zip.ZipOutputStream

/** Redacts diagnostics only in the share copy. Original session logs remain intact on disk. */
object LogRedactor {
    private const val HEX16 = "[0-9A-Fa-f]{16}"

    /** Applied in order: specific Xenia log formats first, then generic key/value forms. */
    private val rules: List<Pair<Regex, String>> = listOf(
        Regex("(?i)/(?:storage/(?:emulated/\\d+|[\\w.-]+)|sdcard|data/user/\\d+)/[^\\s\"'<>]+") to "[storage-path]",
        Regex("(?i)[A-Z]:\\\\Users\\\\[^\\s\"']+") to "[user-path]",
        Regex("""(?i)["']?\b(?:access[_-]?token|refresh[_-]?token|auth[_-]?token|password|authorization|api[_-]?key)\b["']?\s*[:=]\s*(?:"[^"]*"|'[^']*'|[^\s,;]+)""") to "[credential]",
        Regex("(?i)\\bBearer\\s+[A-Za-z0-9._~-]{8,}") to "[credential]",
        // profile_manager.cc: "Loaded {gamertag} (GUID: {xuid}) to slot {n}".
        Regex("\\bLoaded .+? \\(GUID: (?:0x)?$HEX16\\)") to "Loaded [identity] (GUID: [xuid])",
        // user_profile.cc: "User {name} (XUID: {xuid}) ...".
        Regex("\\bUser .+? \\(XUID: (?:0x)?$HEX16\\)") to "User [identity] (XUID: [xuid])",
        // profile_manager.cc account/profile listing lines name the XUID bare.
        Regex("\\b(Loading Account:|Adding profile|Reloading GPDs for profile|Profile) (?:0x)?$HEX16\\b") to "$1 [xuid]",
        // Keys that END in xuid too (logged_profile_slot_0_xuid = "..." in the config dump).
        Regex("""(?i)(["']?\b\w*(?:xuid|gamertag|username|profile[_-]?name)\w*\b["']?\s*[:=]).*$""") to "$1 [identity]",
        Regex("(?i)\\b(?:xuid|guid)\\b\\s*[:=]?\\s*(?:0x)?$HEX16\\b") to "[xuid]",
        Regex("(?i)(?<=content[/\\\\])[a-f0-9]{16}(?=[/\\\\])") to "[xuid]",
        Regex("(?<![\\w.])(?:\\d{1,3}\\.){3}\\d{1,3}(?![\\w.])") to "[ip]",
        // MAC addresses, then IPv6: 4 or more colon-separated groups (a trailing ":port" or ":"
        // included), or shortened by "::". Not 3 groups: logcat's clock times (13:43:48.620).
        Regex("(?i)(?<![\\w:])[a-f0-9]{2}(?::[a-f0-9]{2}){5}(?!\\w|:[a-f0-9])") to "[mac]",
        Regex("(?i)(?<![\\w:])(?:[a-f0-9]{1,4}:){3,}[a-f0-9]{0,4}(?![\\w:])") to "[ip]",
        Regex("(?i)(?<![\\w:])(?:[a-f0-9]{1,4}(?::[a-f0-9]{1,4}){0,6}::(?:[a-f0-9]{1,4}(?::[a-f0-9]{1,4}){0,6})?|::[a-f0-9]{1,4}(?::[a-f0-9]{1,4}){0,6})(?!\\w|:[a-f0-9])") to "[ip]",
        Regex("(?i)\\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\\.[A-Z]{2,}\\b") to "[email]",
    )

    fun redact(line: String): String = rules.fold(line) { text, (pattern, replacement) -> pattern.replace(text, replacement) }
}

/** Flat, bounded ZIP: no nested raw session archives or binary crash dumps in shared diagnostics. */
object SanitizedSessionExport {
    private const val MAX_ARCHIVES = 16
    private const val MAX_ARCHIVE_BYTES = 64L * 1024 * 1024
    private const val MAX_ENTRY_CHARS = 4L * 1024 * 1024
    private const val MAX_TOTAL_CHARS = 32L * 1024 * 1024
    private const val MAX_LINE_CHARS = 8192
    /** The start an xe.log too long to share whole keeps ahead of its end: bytes of a file,
     *  characters of an archived entry (the same for the mostly ASCII logs). */
    internal const val HEAD_SIZE = 256 * 1024
    /** How far an archived entry is read looking for its end; sessions shelve 64 MB at most. */
    private const val MAX_ARCHIVED_ENTRY_CHARS = 96L * 1024 * 1024
    private const val MIDDLE_OMITTED = "[log middle omitted]\n"
    private val allowedName = Regex("(?:(?:previous-[0-9-]+-)?(?:xe\\.log|logcat\\.txt)|logcat-current\\.txt|context\\.json|exit-info\\.txt|exit-trace-\\d+\\.txt)")

    fun create(
        destination: File,
        histories: List<File>,
        current: List<Pair<File, String>>,
    ): File? {
        if (histories.isEmpty() && current.isEmpty()) return null
        destination.parentFile?.mkdirs()
        val temporary = File(destination.parentFile, ".${destination.name}.tmp")
        var written = 0
        var remaining = MAX_TOTAL_CHARS
        try {
            ZipOutputStream(temporary.outputStream().buffered()).use { zip ->
                histories.takeLast(MAX_ARCHIVES).forEach { archive ->
                    if (!archive.isFile || archive.length() > MAX_ARCHIVE_BYTES ||
                        !archive.name.matches(Regex("session_[A-Za-z0-9_-]+\\.zip"))) return@forEach
                    ZipInputStream(FileInputStream(archive).buffered()).use { source ->
                        var entries = 0
                        while (entries++ < 12 && remaining > 0) {
                            val entry = source.nextEntry ?: break
                            // Skip, don't stop: a binary tombstone may precede exit-info.txt.
                            // The archive is size-capped above and the entry count bounds the work.
                            if (entry.isDirectory || !allowedName.matches(entry.name)) continue
                            zip.putNextEntry(ZipEntry("sessions/${archive.nameWithoutExtension}/${entry.name}"))
                            val (text, complete) = archivedText(source, minOf(remaining, MAX_ENTRY_CHARS).toInt(),
                                keepStart = entry.name.endsWith("xe.log"))
                            writeRedacted(text.byteInputStream(), zip, Long.MAX_VALUE)
                            remaining -= text.length
                            zip.closeEntry()
                            written++
                            // nextEntry drains the previous entry. Stop instead when one was too
                            // long to read to its end: don't decompress the rest of a potentially
                            // hostile ZIP.
                            if (!complete) break
                        }
                    }
                }
                current.forEach { (file, name) ->
                    if (!file.isFile || !allowedName.matches(name) || remaining <= 0) return@forEach
                    zip.putNextEntry(ZipEntry("current/$name"))
                    var limit = minOf(remaining, MAX_ENTRY_CHARS)
                    // An xe.log too long to share whole keeps its start too: it opens with the
                    // config, the GPU and driver and the game's own settings, which a verbose
                    // run (kernel calls logged) pushes out of the end. Measured in bytes like
                    // the end's skip, so the end never repeats a line of it.
                    if (name.endsWith("xe.log") && file.length() > limit && limit > 2 * HEAD_SIZE) {
                        val start = startOf(file)
                        writeRedacted(start.inputStream(), zip, Long.MAX_VALUE)
                        zip.write(MIDDLE_OMITTED.toByteArray(Charsets.UTF_8))
                        remaining -= start.size
                        limit -= start.size
                    }
                    FileInputStream(file).use { source ->
                        val skip = (file.length() - limit).coerceAtLeast(0)
                        if (skip > 0) {
                            source.skip(skip)
                            // Tail snapshots can begin mid-line; drop it rather than leaking a
                            // token suffix that no complete-line pattern could recognize.
                            while (source.read().let { it >= 0 && it != '\n'.code }) { }
                        }
                        remaining -= writeRedacted(source, zip, limit)
                    }
                    zip.closeEntry()
                    written++
                }
            }
            if (written == 0) return null
            if (!temporary.renameTo(destination)) error("Could not finalize diagnostics")
            return destination
        } finally {
            temporary.delete()
        }
    }

    /** A file's first [HEAD_SIZE] bytes, up to their last whole line. */
    private fun startOf(file: File): ByteArray {
        val bytes = ByteArray(HEAD_SIZE)
        var count = 0
        FileInputStream(file).use { input ->
            while (count < bytes.size) {
                val n = input.read(bytes, count, bytes.size - count)
                if (n < 0) break
                count += n
            }
        }
        while (count > 0 && bytes[count - 1] != '\n'.code.toByte()) count--
        return bytes.copyOf(count)
    }

    /**
     * An archived entry's text in at most [limit] characters: whole when it fits, else its end
     * from a whole line, after its start up to a whole line when [keepStart]. A ZIP entry can't
     * seek, so the end is kept in a ring as the entry is read; reading stops at
     * [MAX_ARCHIVED_ENTRY_CHARS], and then the second value is false.
     */
    private fun archivedText(input: InputStream, limit: Int, keepStart: Boolean): Pair<String, Boolean> {
        val reader = InputStreamReader(input, Charsets.UTF_8).buffered()
        val startCap = if (keepStart && limit > 2 * HEAD_SIZE) HEAD_SIZE else 0
        val start = StringBuilder()
        val ring = CharArray(limit - startCap)
        var first = 0
        var size = 0
        var dropped = false
        var complete = true
        var read = 0L
        while (true) {
            val c = reader.read()
            if (c < 0) break
            if (++read > MAX_ARCHIVED_ENTRY_CHARS) { complete = false; dropped = true; break }
            when {
                start.length < startCap -> start.append(c.toChar())
                size < ring.size -> ring[(first + size++) % ring.size] = c.toChar()
                else -> { ring[first] = c.toChar(); first = (first + 1) % ring.size; dropped = true }
            }
        }
        val end = buildString(size) { for (i in 0 until size) append(ring[(first + i) % ring.size]) }
        if (!dropped) return (start.toString() + end) to complete
        // Both sides of the cut end at whole lines: a part of a line could keep part of a value
        // that no pattern recognizes any more.
        val newline = end.indexOf('\n')
        val keptEnd = if (newline < 0) "" else end.substring(newline + 1)
        if (startCap == 0) return keptEnd to complete
        return (start.substring(0, start.lastIndexOf('\n') + 1) + MIDDLE_OMITTED + keptEnd) to complete
    }

    /** Read one character at a time from a buffered reader to bound long lines and ZIP bombs. */
    private fun writeRedacted(input: InputStream, zip: ZipOutputStream, limit: Long): Long {
        val reader = InputStreamReader(input, Charsets.UTF_8).buffered()
        val line = StringBuilder()
        var dropping = false
        var count = 0L
        while (count < limit) {
            val c = reader.read()
            if (c < 0) break
            count++
            if (c == '\n'.code) {
                val text = if (dropping) "[long line omitted]" else LogRedactor.redact(line.toString())
                zip.write("$text\n".toByteArray(Charsets.UTF_8))
                line.setLength(0)
                dropping = false
            } else if (!dropping && line.length < MAX_LINE_CHARS) {
                line.append(c.toChar())
            } else {
                dropping = true
            }
        }
        if (dropping) zip.write("[long line omitted]\n".toByteArray(Charsets.UTF_8))
        else if (line.isNotEmpty()) zip.write((LogRedactor.redact(line.toString()) + "\n").toByteArray(Charsets.UTF_8))
        if (count == limit) zip.write("[diagnostics truncated]\n".toByteArray(Charsets.UTF_8))
        return count
    }
}
