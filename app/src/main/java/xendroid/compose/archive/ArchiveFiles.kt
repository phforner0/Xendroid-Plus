package xendroid.compose.archive

import java.io.File
import java.io.FileOutputStream
import java.io.InputStream
import java.io.RandomAccessFile
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.SimpleFileVisitor
import java.nio.file.FileVisitResult
import java.nio.file.StandardCopyOption
import java.nio.file.attribute.BasicFileAttributes
import java.security.MessageDigest
import java.util.Locale
import java.util.zip.CRC32
import java.util.zip.ZipFile

data class ArchiveLimits(
    val packedBytes: Long = 512L * 1024 * 1024,
    val totalBytes: Long = 1024L * 1024 * 1024,
    val entryBytes: Long = 256L * 1024 * 1024,
    val entries: Int = 20_000,
)

object ArchiveFiles {
    fun relative(name: String): String {
        require(name.isNotEmpty() && name.length <= 1024 && !name.contains(':') && !name.contains('\u0000')) {
            "Invalid archive path"
        }
        val normalized = name.replace('\\', '/')
        require(!normalized.startsWith('/')) { "Invalid archive path" }
        var clean = normalized.trimEnd('/')
        while (clean.startsWith("./")) {
            clean = clean.removePrefix("./")
        }
        val parts = clean.split('/')
        require(parts.size <= 64) { "Archive path is too deep" }
        require(parts.all { it.isNotEmpty() && it != "." && it != ".." }) { "Invalid archive path" }
        return clean
    }

    fun resolve(root: File, name: String): File {
        val clean = relative(name)
        val base = root.canonicalFile
        var current = root
        require(!Files.isSymbolicLink(root.toPath())) { "Symbolic link root" }
        for (part in clean.split('/')) {
            current = File(current, part)
            require(!Files.isSymbolicLink(current.toPath())) { "Symbolic link in destination" }
        }
        val result = current.canonicalFile
        require(result.toPath().startsWith(base.toPath()) && result != base) { "Archive path escaped root" }
        return result
    }

    fun sha256(file: File): String = file.inputStream().use { sha256(it) }
    fun sha256(input: InputStream): String {
        val digest = MessageDigest.getInstance("SHA-256")
        val buffer = ByteArray(64 * 1024)
        while (true) {
            val n = input.read(buffer)
            if (n < 0) break
            digest.update(buffer, 0, n)
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }

    /** Deletion never follows links, even in a partly built transaction. */
    fun deleteTree(file: File) {
        if (!Files.exists(file.toPath(), java.nio.file.LinkOption.NOFOLLOW_LINKS)) return
        Files.walkFileTree(file.toPath(), object : SimpleFileVisitor<Path>() {
            override fun visitFile(path: Path, attrs: BasicFileAttributes): FileVisitResult {
                Files.delete(path); return FileVisitResult.CONTINUE
            }
            override fun postVisitDirectory(path: Path, error: java.io.IOException?): FileVisitResult {
                if (error != null) throw error
                Files.delete(path); return FileVisitResult.CONTINUE
            }
        })
    }

    fun atomicText(file: File, text: String) = atomicBytes(file, text.toByteArray())

    /** Readers see the old or the new content, never a truncated file. */
    fun atomicBytes(file: File, bytes: ByteArray) {
        check(file.parentFile!!.isDirectory || file.parentFile!!.mkdirs())
        val temp = File.createTempFile(".${file.name}.", ".tmp", file.parentFile)
        try {
            FileOutputStream(temp).use { it.write(bytes); it.fd.sync() }
            Files.move(temp.toPath(), file.toPath(), StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING)
        } finally { temp.delete() }
    }

    /** Reads at most [limit] bytes; a longer stream is rejected, not truncated. */
    fun readBounded(input: InputStream, limit: Long): ByteArray {
        val out = java.io.ByteArrayOutputStream()
        val buffer = ByteArray(64 * 1024)
        var total = 0L
        while (true) {
            val n = input.read(buffer)
            if (n < 0) break
            total += n
            require(total <= limit) { "File exceeds the ${limit / (1024 * 1024)} MB limit" }
            out.write(buffer, 0, n)
        }
        return out.toByteArray()
    }

    /** Extract into a NEW private staging directory. Files are always ordinary files,
     * never symlinks or devices; CRC, names, count and actual decoded bytes are checked. */
    fun unpack(archive: File, staging: File, limits: ArchiveLimits, checkCancelled: () -> Unit = {}): List<File> {
        require(archive.isFile && archive.length() <= limits.packedBytes) { "Archive is too large" }
        require(staging.isDirectory && staging.list().orEmpty().isEmpty()) { "Staging must be empty" }
        val seen = mutableSetOf<String>()
        val result = mutableListOf<File>()
        var total = 0L
        ZipFile(archive).use { zip ->
            require(zip.size() <= limits.entries) { "Too many archive entries" }
            val entries = zip.entries()
            while (entries.hasMoreElements()) {
                checkCancelled()
                val entry = entries.nextElement()
                val isDir = entry.isDirectory || entry.name.endsWith("/") || entry.name.endsWith("\\")
                if (isDir) {
                    val raw = entry.name.replace('\\', '/')
                    if (!raw.startsWith('/')) {
                        var dirName = raw.trimEnd('/')
                        while (dirName.startsWith("./")) {
                            dirName = dirName.removePrefix("./")
                        }
                        if (dirName.isEmpty() || dirName == ".") continue
                    }
                }
                val name = relative(entry.name)
                require(seen.add(name.lowercase(Locale.ROOT))) { "Duplicate archive path" }
                val dest = resolve(staging, name)
                if (isDir) {
                    check(dest.isDirectory || dest.mkdirs()); continue
                }
                require(entry.size in 0..limits.entryBytes && total + entry.size <= limits.totalBytes) { "Archive exceeds size limit" }
                check(dest.parentFile!!.isDirectory || dest.parentFile!!.mkdirs())
                val crc = CRC32()
                var bytes = 0L
                zip.getInputStream(entry).use { input ->
                    FileOutputStream(dest).use { output ->
                        val buffer = ByteArray(64 * 1024)
                        while (true) {
                            checkCancelled()
                            val n = input.read(buffer)
                            if (n < 0) break
                            bytes += n; total += n
                            require(bytes <= limits.entryBytes && total <= limits.totalBytes && bytes <= entry.size) { "Archive exceeds size limit" }
                            crc.update(buffer, 0, n); output.write(buffer, 0, n)
                        }
                        output.fd.sync()
                    }
                }
                require(bytes == entry.size && crc.value == entry.crc) { "Corrupt archive entry" }
                result.add(dest)
            }
        }
        return result
    }
}

private val directoryMonitors = java.util.concurrent.ConcurrentHashMap<String, Any>()

/**
 * Serializes read-modify-write of the small files in [dir] across threads (a monitor:
 * one JVM cannot hold two locks on a file) and processes (a file lock).
 */
fun <T> withDirectoryLock(dir: File, block: () -> T): T {
    check(dir.isDirectory || dir.mkdirs()) { "Cannot create ${dir.name}" }
    return synchronized(directoryMonitors.computeIfAbsent(dir.canonicalPath) { Any() }) {
        RandomAccessFile(File(dir, ".lock"), "rw").use { file -> file.channel.lock().use { block() } }
    }
}

/** Storage operations and a running guest are mutually exclusive across processes. */
class ContentLease private constructor(
    val root: File,
    private val file: RandomAccessFile,
    private val key: String,
) : AutoCloseable {
    private val lock = file.channel.tryLock() ?: error("Content is in use")
    private val closed = java.util.concurrent.atomic.AtomicBoolean()

    override fun close() {
        if (!closed.compareAndSet(false, true)) return
        try { lock.release() } finally { try { file.close() } finally { held.remove(key) } }
    }

    companion object {
        /**
         * Lock files this process holds a lease on. A second lease in the same process must fail
         * BEFORE opening the file: closing any descriptor of a file drops every POSIX lock the
         * process holds on it, so the failed attempt's close would silently free the first
         * lease for the other process (reproduced on Linux; see FileLock's docs).
         */
        private val held: MutableSet<String> = java.util.concurrent.ConcurrentHashMap.newKeySet()

        fun acquire(root: File): ContentLease {
            check(root.isDirectory || root.mkdirs())
            require(!Files.isSymbolicLink(root.toPath()))
            val lockPath = ArchiveFiles.resolve(root, ".content-session.lock")
            val key = lockPath.canonicalPath
            if (!held.add(key)) throw ContentBusyException()
            val file = try { RandomAccessFile(lockPath, "rw") } catch (e: Exception) { held.remove(key); throw e }
            return try { ContentLease(root.canonicalFile, file, key) } catch (e: Exception) {
                file.close(); held.remove(key); throw ContentBusyException(e)
            }
        }
    }
}

/** The lease is held by another operation: a running game or a save/content job. */
class ContentBusyException(cause: Throwable? = null) :
    IllegalStateException("Close the running game or wait for the content operation", cause)

/**
 * Waits a bounded time for a short storage job (an automatic backup, a profile edit)
 * to release the lease instead of failing a game launch outright. Only
 * [ContentBusyException] is retried; any other failure is final.
 */
suspend fun <T> acquireWithRetry(
    timeoutMs: Long,
    pollMs: Long = 250,
    now: () -> Long = System::currentTimeMillis,
    sleep: suspend (Long) -> Unit = { kotlinx.coroutines.delay(it) },
    onBusy: () -> Unit = {},
    acquire: () -> T,
): T {
    val deadline = now() + timeoutMs
    var notified = false
    while (true) {
        try {
            return acquire()
        } catch (e: ContentBusyException) {
            if (now() >= deadline) throw e
            if (!notified) { notified = true; onBusy() }
            sleep(pollMs.coerceAtMost((deadline - now()).coerceAtLeast(1)))
        }
    }
}
