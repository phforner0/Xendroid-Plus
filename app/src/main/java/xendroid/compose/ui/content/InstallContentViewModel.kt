package xendroid.compose.ui.content

import android.content.Context
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.firstOrNull
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import xendroid.compose.core.ContentPaths
import xendroid.compose.core.StorageAccess
import xendroid.compose.core.EmulatorRuntime
import xendroid.compose.core.GameMetadataSource
import xendroid.compose.data.PreferencesStore
import java.io.File

/**
 * Global, install-only content entrypoint. Accepts ANY content package (DLC, title
 * update, profile, arcade game, …) without pre-selecting a game — native install_content
 * reads the package header and places it. No list, no delete (see ContentManagerViewModel
 * for the per-game manager). Shares ContentInstallState + ContentInstallDialogs.
 */
class InstallContentViewModel(
    private val appContext: Context,
    private val metadata: GameMetadataSource,
    private val prefs: PreferencesStore,
) : ViewModel() {

    private val _state = MutableStateFlow<ContentInstallState>(ContentInstallState.Idle)
    val state: StateFlow<ContentInstallState> = _state.asStateFlow()

    fun dismiss() { _state.value = ContentInstallState.Idle }

    /** [srcPath] = absolute host path to the picked package. */
    fun install(srcPath: String) = viewModelScope.launch {
        _state.value = ContentInstallState.Busy(appContext.getString(xendroid.compose.R.string.cm_preparing))
        val pre = withContext(Dispatchers.IO) { validate(srcPath) }
        when (pre) {
            is PreCheck.Reject -> _state.value = ContentInstallState.Failed(pre.message)
            is PreCheck.Done -> _state.value = ContentInstallState.Done(pre.message)
            is PreCheck.Overwrite ->
                _state.value = ContentInstallState.ConfirmOverwrite(srcPath, pre.name)
            is PreCheck.Copy -> runCopy(srcPath, pre.name, pre.gamesDir)
            is PreCheck.Disc -> runDiscInstall(srcPath, pre.items)
            is PreCheck.Ok -> runInstall(srcPath, pre.name, pre.label)
        }
    }

    fun confirmOverwrite(srcPath: String, name: String) = viewModelScope.launch {
        val header = withContext(Dispatchers.IO) {
            EmulatorRuntime.ensureLoaded(); metadata.readContentHeader(srcPath)
        }
        if (header == null) {
            _state.value = ContentInstallState.Failed(appContext.getString(xendroid.compose.R.string.ci_reread_failed))
            return@launch
        }
        // A full game overwrites the container in the games folder; add-on content
        // overwrites its native content-tree placement. This path skips validate(), so
        // each branch re-checks free space against the volume it writes to.
        if (ContentPaths.isLaunchableGameType(header.contentType)) {
            val gamesDir = gamesDirOrNull()
                ?: run { _state.value = ContentInstallState.Failed(appContext.getString(xendroid.compose.R.string.ci_no_games_folder)); return@launch }
            storageShortfallOn(appContext, gamesDir, File(srcPath).length())?.let {
                _state.value = ContentInstallState.Failed(it); return@launch
            }
            runCopy(srcPath, name, gamesDir)
        } else {
            storageShortfall(appContext, header.contentSize)?.let {
                _state.value = ContentInstallState.Failed(it); return@launch
            }
            runInstall(srcPath, name, ContentPaths.contentTypeLabel(header.contentType))
        }
    }

    private sealed interface PreCheck {
        data class Ok(val name: String, val label: String) : PreCheck
        /** A disc image carrying installable packages (a mandatory-install title). */
        data class Disc(val items: List<GameMetadataSource.DiscContent>) : PreCheck
        data class Overwrite(val name: String) : PreCheck
        /** Full game to copy into [gamesDir] (no existing container in the way). */
        data class Copy(val name: String, val gamesDir: File) : PreCheck
        /** No copy needed (already in the games folder). */
        data class Done(val message: String) : PreCheck
        data class Reject(val message: String) : PreCheck
    }

    private suspend fun validate(srcPath: String): PreCheck {
        EmulatorRuntime.ensureLoaded()
        val src = File(srcPath)
        if (!src.isFile) return PreCheck.Reject(appContext.getString(xendroid.compose.R.string.ci_open_failed))
        val meta = metadata.readContentHeader(srcPath)
            ?: return validateDisc(srcPath)
        val name = meta.displayName.ifBlank { src.name }
        // A full game (XBLA/arcade, GoD, ...) is COPIED into the games folder as a
        // container so the library boots it; add-on content is installed natively below.
        if (ContentPaths.isLaunchableGameType(meta.contentType)) return validateGameCopy(src, name)
        storageShortfall(appContext, meta.contentSize)?.let { return PreCheck.Reject(it) }
        val label = ContentPaths.contentTypeLabel(meta.contentType)
        // Profiles aren't placed under the machine XUID/title tree, so skip the
        // overwrite pre-check and install directly. Everything else is checked.
        if (meta.contentType == ContentPaths.PROFILE_CONTENT_TYPE)
            return PreCheck.Ok(name, label)
        if (meta.titleId == null) return PreCheck.Ok(name, label)
        val pkgDir = File(ContentPaths.contentDir(meta.titleId, meta.contentType), meta.packageName)
        return if (pkgDir.exists()) PreCheck.Overwrite(name) else PreCheck.Ok(name, label)
    }

    /** Decide the full-game copy: needs a games folder; no-op if already there; confirm a clobber. */
    private suspend fun validateGameCopy(src: File, name: String): PreCheck {
        val gamesDir = gamesDirOrNull() ?: return PreCheck.Reject(appContext.getString(xendroid.compose.R.string.ci_no_games_folder))
        val dest = File(gamesDir, src.name)
        // Already inside any of the library's folders (L03), or the very file it would become.
        if (allGamesDirs().any { src.canonicalPath.startsWith(it.canonicalPath + File.separator) } ||
            src.canonicalFile == dest.canonicalFile
        ) {
            return PreCheck.Done(appContext.getString(xendroid.compose.R.string.ci_already_in_folder, name))
        }
        storageShortfallOn(appContext, gamesDir, src.length())?.let { return PreCheck.Reject(it) }
        return if (dest.exists()) PreCheck.Overwrite(name) else PreCheck.Copy(name, gamesDir)
    }

    /** A picked file that is not a package may still be a disc image whose \content\ tree
     *  holds the install payload; anything else is rejected as before. */
    private fun validateDisc(srcPath: String): PreCheck {
        val items = metadata.listDiscContent(srcPath)
        if (items.isEmpty()) {
            return PreCheck.Reject(appContext.getString(xendroid.compose.R.string.ci_not_package))
        }
        storageShortfall(appContext, items.sumOf { it.size })?.let { return PreCheck.Reject(it) }
        return PreCheck.Disc(items)
    }

    /** Installs every package on the disc, one at a time. Each is staged out of the image
     *  natively, so progress is reported per package rather than across the whole set. */
    private suspend fun runDiscInstall(
        discPath: String,
        items: List<GameMetadataSource.DiscContent>,
    ) {
        val scratch = File(appContext.cacheDir, "disc-install")
        var installed = 0
        for ((index, item) in items.withIndex()) {
            _state.value = ContentInstallState.Busy(
                appContext.getString(xendroid.compose.R.string.ci_installing_n, index + 1, items.size), 0f)
            val poll = viewModelScope.launch {
                while (isActive) {
                    val p = EmulatorRuntime.emulator?.installProgress() ?: 0f
                    (_state.value as? ContentInstallState.Busy)
                        ?.let { _state.value = it.copy(progress = p.coerceIn(0f, 1f)) }
                    delay(200)
                }
            }
            val status = withContext(Dispatchers.IO) {
                val emu = EmulatorRuntime.emulator ?: return@withContext -1
                runCatching { StorageAccess.acquire().use { emu.install_disc_content(
                    discPath, item.innerPath,
                    ContentPaths.contentRoot().absolutePath, scratch.absolutePath) } }.getOrDefault(0xC0000022.toInt())
            }
            poll.cancel()
            if (status != 0) {
                _state.value = ContentInstallState.Failed(
                    appContext.getString(xendroid.compose.R.string.ci_installed_partial, installed, items.size) + " " + installReasonFor(appContext, status))
                return
            }
            installed++
        }
        runCatching { scratch.deleteRecursively() }
        _state.value = ContentInstallState.Done(
            appContext.resources.getQuantityString(xendroid.compose.R.plurals.ci_installed_disc, installed, installed))
    }

    /** Full games are copied into the library's first folder (L03: there can be several). */
    private suspend fun gamesDirOrNull(): File? = allGamesDirs().firstOrNull()

    private suspend fun allGamesDirs(): List<File> =
        prefs.gameDirPaths.firstOrNull().orEmpty().filter { it.isNotBlank() }.map { File(it) }

    private suspend fun runInstall(srcPath: String, name: String, label: String?) {
        val installing = appContext.getString(xendroid.compose.R.string.ci_installing)
        _state.value = ContentInstallState.Busy(installing, 0f)
        // Poll native install progress while the VFS walk blocks an IO thread (the
        // getter reads file-static atomics, so concurrent reads are safe).
        val poll = viewModelScope.launch {
            while (isActive) {
                val p = EmulatorRuntime.emulator?.installProgress() ?: 0f
                (_state.value as? ContentInstallState.Busy)
                    ?.takeIf { it.message == installing }
                    ?.let { _state.value = it.copy(progress = p.coerceIn(0f, 1f)) }
                delay(200)
            }
        }
        val status = withContext(Dispatchers.IO) {
            val emu = EmulatorRuntime.emulator ?: return@withContext -1
            runCatching { StorageAccess.acquire().use {
                emu.install_content(srcPath, ContentPaths.contentRoot().absolutePath)
            } }.getOrDefault(0xC0000022.toInt())
        }
        poll.cancel()
        if (status == 0) {
            val what = label?.let { "$it “$name”" } ?: "“$name”"
            _state.value = ContentInstallState.Done(appContext.getString(xendroid.compose.R.string.ci_installed_any, what))
        } else {
            _state.value = ContentInstallState.Failed(installReasonFor(appContext, status))
        }
    }

    /** Copy a full-game container into the games folder so the library scan picks it up.
     *  Writes to a temp file in the SAME dir then renames, so a partial copy never looks
     *  like a game. These containers are large -> indeterminate Busy. */
    private suspend fun runCopy(srcPath: String, name: String, gamesDir: File) {
        _state.value = ContentInstallState.Busy(appContext.getString(xendroid.compose.R.string.ci_copying))
        val ok = withContext(Dispatchers.IO) {
            runCatching {
                val src = File(srcPath)
                val dest = File(gamesDir, src.name)
                val tmp = File(gamesDir, ".${src.name}.part")
                tmp.delete()
                src.inputStream().use { input -> tmp.outputStream().use { input.copyTo(it) } }
                if (!tmp.renameTo(dest)) {
                    dest.delete()
                    if (!tmp.renameTo(dest)) { tmp.delete(); return@runCatching false }
                }
                true
            }.getOrDefault(false)
        }
        _state.value = if (ok) {
            ContentInstallState.Done(appContext.getString(xendroid.compose.R.string.ci_copied, name))
        } else {
            ContentInstallState.Failed(appContext.getString(xendroid.compose.R.string.ci_copy_failed))
        }
    }


}
