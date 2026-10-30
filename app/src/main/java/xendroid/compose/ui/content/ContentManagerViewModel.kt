package xendroid.compose.ui.content

import android.content.Context
import android.os.Environment
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import xendroid.compose.Emulator
import xendroid.compose.core.ContentPaths
import xendroid.compose.core.ContentVersion
import xendroid.compose.core.EmulatorRuntime
import xendroid.compose.core.GameMetadataSource
import xendroid.compose.core.StorageAccess
import xendroid.compose.saves.ContentTrash
import xendroid.compose.saves.TrashFullException
import xendroid.compose.saves.TrashedContent
import java.io.File

/**
 * Installed DLC and title updates, the content trash and installing packages: one game's
 * ([titleId]), or every game's (null: the Content area), where the list is grouped by game.
 */
class ContentManagerViewModel(
    private val appContext: Context,
    private val metadata: GameMetadataSource,
    private val titleId: String?,
    /** Where downloaded packages wait (Downloads); null when it cannot be read. */
    private val inbox: () -> File? = { Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS) },
    /** Round 2: the player's content folders, searched with their subfolders. */
    private val folders: () -> List<String> = { ContentFolders.read(appContext) },
) : ViewModel() {

    data class ContentEntry(
        val pkgDir: String,
        val displayName: String,
        val size: Long,
        val contentType: Int,
        /** The title it is installed for (uppercase). */
        val titleId: String = "",
        /** Round 2: a title update's version, from its patch executable ("1.0.3.0"). */
        val version: String? = null,
    )

    /** One game's installed packages. */
    data class GameContent(val titleId: String, val dlc: List<ContentEntry>, val updates: List<ContentEntry>) {
        val packages: Int get() = dlc.size + updates.size
        val bytes: Long get() = dlc.sumOf { it.size } + updates.sumOf { it.size }
    }

    sealed interface ListState {
        data object Loading : ListState
        /** [trashed]: the game's packages in the trash (L12), every game's in the Content area;
         *  [trashUsed] counts every game's. [games]: by game, those with something installed. */
        data class Loaded(
            val dlc: List<ContentEntry>,
            val updates: List<ContentEntry>,
            val trashed: List<TrashedContent> = emptyList(),
            val trashUsed: Long = 0,
            val trashQuota: Long = ContentTrash.DEFAULT_QUOTA,
            val games: List<GameContent> = emptyList(),
        ) : ListState
        data class Error(val message: String) : ListState
    }

    /** True for the Content area (every game's content). */
    val allGames: Boolean get() = titleId == null

    private val _found = MutableStateFlow<List<FoundPackage>?>(null)
    /** Packages lying in Downloads (this game's only, for one game); null until looked for. */
    val found: StateFlow<List<FoundPackage>?> = _found.asStateFlow()

    /** Round 2: a content folder and how many packages the last look found under it. */
    data class FolderCount(val path: String, val packages: Int, val readable: Boolean)

    private val _folderCounts = MutableStateFlow<List<FolderCount>>(emptyList())
    val folderCounts: StateFlow<List<FolderCount>> = _folderCounts.asStateFlow()

    private val _freeBytes = MutableStateFlow<Long?>(null)
    /** Free space where packages are installed. */
    val freeBytes: StateFlow<Long?> = _freeBytes.asStateFlow()

    /** L12: removing moves a package to the trash; restore or delete it for good from there. */
    private val trash by lazy { ContentTrash(ContentPaths.contentRoot()) }

    private val _listState = MutableStateFlow<ListState>(ListState.Loading)
    val listState: StateFlow<ListState> = _listState.asStateFlow()

    /** Install/overwrite/progress/result reuse the shared ContentInstallState. */
    private val _state = MutableStateFlow<ContentInstallState>(ContentInstallState.Idle)
    val state: StateFlow<ContentInstallState> = _state.asStateFlow()

    /** Delete confirmation stays per-game (separate from the shared install flow). */
    sealed interface DeleteState {
        data object Idle : DeleteState
        data class Confirm(val item: ContentEntry) : DeleteState
        /** The trash has no room for [item]: empty it, or delete [item] for good. */
        data class TrashFull(val item: ContentEntry, val used: Long, val quota: Long) : DeleteState
        data class ConfirmPurge(val entry: TrashedContent) : DeleteState
        data object ConfirmEmptyTrash : DeleteState
    }

    private val _deleteState = MutableStateFlow<DeleteState>(DeleteState.Idle)
    val deleteState: StateFlow<DeleteState> = _deleteState.asStateFlow()

    init { refresh() }

    fun dismiss() {
        _state.value = ContentInstallState.Idle
        _deleteState.value = DeleteState.Idle
    }

    fun refresh() = viewModelScope.launch {
        _listState.value = ListState.Loading
        _listState.value = withContext(Dispatchers.IO) {
            EmulatorRuntime.ensureLoaded()
            val emu = EmulatorRuntime.emulator
                ?: return@withContext ListState.Error(appContext.getString(xendroid.compose.R.string.pf_no_emulator))
            // Settle a removal or restore a killed process left half done, before listing.
            runCatching { StorageAccess.acquire().use { trash.recover(it) } }
            val contentRoot = ContentPaths.contentRoot()
            val root = contentRoot.absolutePath
            try {
                // The core lists one title at a time: the game, or every title with a content folder.
                val titles = titleId?.let { listOf(it.uppercase()) } ?: ContentCatalog.titlesWithContent(contentRoot)
                val games = titles.map { title ->
                    val dlc = emu.list_content(root, title, ContentPaths.DLC_CONTENT_TYPE)
                        ?: return@withContext ListState.Error(appContext.getString(xendroid.compose.R.string.cm_read_failed))
                    val updates = emu.list_content(root, title, ContentPaths.TU_CONTENT_TYPE)
                        ?: return@withContext ListState.Error(appContext.getString(xendroid.compose.R.string.cm_read_failed))
                    GameContent(title, dlc.toEntries(ContentPaths.DLC_CONTENT_TYPE, title), updates.toEntries(ContentPaths.TU_CONTENT_TYPE, title))
                }.filter { titleId != null || it.packages > 0 }
                val trashed = runCatching { trash.list() }.getOrDefault(emptyList()).sortedByDescending { it.deletedAt }
                ListState.Loaded(
                    dlc = games.flatMap { it.dlc },
                    updates = games.flatMap { it.updates },
                    trashed = if (titleId == null) trashed else trashed.filter { it.titleId.equals(titleId, ignoreCase = true) },
                    trashUsed = trashed.sumOf { it.bytes },
                    trashQuota = trash.quotaBytes,
                    games = games,
                )
            } catch (t: RuntimeException) {
                ListState.Error(t.message ?: appContext.getString(xendroid.compose.R.string.cm_read_failed))
            }
        }
        _freeBytes.value = withContext(Dispatchers.IO) { ContentCatalog.freeBytes() }
    }

    /** Looks for packages: in Downloads (a few header bytes of its newest files) and, round 2,
     *  in the content folders and every folder inside them. The core reads the header of those
     *  that look like one; each says whether it is installed already and a title update's version. */
    fun lookForPackages() = viewModelScope.launch {
        val (found, counts) = withContext(Dispatchers.IO) {
            EmulatorRuntime.ensureLoaded()
            fun describe(file: File, folder: String?): FoundPackage? {
                val meta = metadata.readContentHeader(file.absolutePath) ?: return null
                val installed = meta.titleId?.let { File(ContentPaths.contentDir(it, meta.contentType), meta.packageName).exists() } == true
                val version = if (meta.contentType == ContentPaths.TU_CONTENT_TYPE) ContentVersion.ofPackage(file) else null
                return FoundPackage(file.absolutePath, file.name, meta.titleId, meta.contentType, meta.displayName.ifBlank { file.name },
                    file.length(), file.lastModified(), folder, installed, version)
            }
            val mine = { p: FoundPackage -> titleId == null || p.titleId.equals(titleId, ignoreCase = true) }
            val fromDownloads = ContentCatalog.packagesIn(runCatching { inbox() }.getOrNull()).mapNotNull { describe(it, null) }
            val counts = mutableListOf<FolderCount>()
            val fromFolders = runCatching { folders() }.getOrDefault(emptyList()).flatMap { path ->
                val dir = File(path)
                val packages = ContentCatalog.packagesUnder(dir).mapNotNull { describe(it, path) }.filter(mine)
                counts += FolderCount(path, packages.size, dir.canRead())
                packages
            }
            (fromDownloads.filter(mine) + fromFolders).distinctBy { it.path } to counts.toList()
        }
        _folderCounts.value = counts
        _found.value = found
    }

    /** Round 2: a folder of title updates and DLC to look in, with its subfolders. */
    fun addFolder(path: String) {
        ContentFolders.add(appContext, path)
        lookForPackages()
    }

    fun removeFolder(path: String) {
        ContentFolders.remove(appContext, path)
        lookForPackages()
    }

    private fun Array<Emulator.ContentItem>.toEntries(contentType: Int, title: String) =
        map {
            val version = if (contentType == ContentPaths.TU_CONTENT_TYPE)
                runCatching { ContentVersion.ofInstalled(File(ContentPaths.contentDir(title, contentType), it.pkgDir)) }.getOrNull() else null
            ContentEntry(it.pkgDir, it.displayName ?: it.pkgDir, it.size, contentType, title, version)
        }.sortedBy { it.displayName.lowercase() }

    /** The title [item] is installed for. */
    private fun titleOf(item: ContentEntry): String = item.titleId.ifEmpty { titleId.orEmpty() }

    /** [srcPath] = absolute host path to the picked package. */
    fun install(srcPath: String) = viewModelScope.launch {
        _state.value = ContentInstallState.Busy(appContext.getString(xendroid.compose.R.string.cm_preparing))
        val pre = withContext(Dispatchers.IO) { validate(srcPath) }
        when (pre) {
            is PreCheck.Reject -> _state.value = ContentInstallState.Failed(pre.message)
            is PreCheck.Overwrite ->
                _state.value = ContentInstallState.ConfirmOverwrite(srcPath, pre.name)
            is PreCheck.Ok -> runInstall(srcPath, pre.name)
        }
    }

    /** User confirmed overwrite of an existing package dir (OD3). */
    fun confirmOverwrite(srcPath: String, name: String) =
        viewModelScope.launch { runInstall(srcPath, name) }

    fun requestDelete(item: ContentEntry) { _deleteState.value = DeleteState.Confirm(item) }

    /** L12: to the trash (restorable). A full trash asks what to do instead of deleting. */
    fun delete(item: ContentEntry) = viewModelScope.launch {
        _deleteState.value = DeleteState.Idle
        _state.value = ContentInstallState.Busy(appContext.getString(xendroid.compose.R.string.cm_trashing))
        val result = withContext(Dispatchers.IO) {
            runCatching {
                StorageAccess.acquire().use { trash.moveToTrash(it, titleOf(item), item.contentType, item.pkgDir, item.displayName) }
            }
        }
        result.onSuccess {
            _state.value = ContentInstallState.Done(appContext.getString(xendroid.compose.R.string.cm_trashed, item.displayName))
            refresh()
        }.onFailure { e ->
            if (e is TrashFullException) {
                _state.value = ContentInstallState.Idle
                _deleteState.value = DeleteState.TrashFull(item, e.usedBytes, e.quotaBytes)
            } else {
                _state.value = ContentInstallState.Failed(
                    if (e is xendroid.compose.archive.ContentBusyException) appContext.getString(xendroid.compose.R.string.cm_close_game) else appContext.getString(xendroid.compose.R.string.cm_trash_failed, e.message))
            }
        }
    }

    fun restore(entry: TrashedContent) = viewModelScope.launch {
        _state.value = ContentInstallState.Busy(appContext.getString(xendroid.compose.R.string.cm_restoring))
        val result = withContext(Dispatchers.IO) {
            runCatching { StorageAccess.acquire().use { trash.restore(it, entry.id) } }
        }
        result.onSuccess {
            _state.value = ContentInstallState.Done(appContext.getString(xendroid.compose.R.string.cm_restored, entry.displayName))
            refresh()
        }.onFailure { e ->
            _state.value = ContentInstallState.Failed(when ((e as? xendroid.compose.saves.RestoreRefusedException)?.why) {
                xendroid.compose.saves.RestoreRefusedException.Why.GONE -> appContext.getString(xendroid.compose.R.string.cm_restore_gone)
                xendroid.compose.saves.RestoreRefusedException.Why.INSTALLED_AGAIN ->
                    appContext.getString(xendroid.compose.R.string.cm_restore_installed_again, (e as xendroid.compose.saves.RestoreRefusedException).name)
                xendroid.compose.saves.RestoreRefusedException.Why.HEADER_IN_USE -> appContext.getString(xendroid.compose.R.string.cm_restore_header_in_use)
                null -> if (e is xendroid.compose.archive.ContentBusyException) appContext.getString(xendroid.compose.R.string.cm_close_game)
                    else e.message ?: appContext.getString(xendroid.compose.R.string.cm_restore_failed)
            })
        }
    }

    fun requestPurge(entry: TrashedContent) { _deleteState.value = DeleteState.ConfirmPurge(entry) }
    fun requestEmptyTrash() { _deleteState.value = DeleteState.ConfirmEmptyTrash }

    /** Deletes trashed packages for good: one, or ([entry] null) the whole trash, every game's. */
    fun purge(entry: TrashedContent?) = viewModelScope.launch {
        _deleteState.value = DeleteState.Idle
        _state.value = ContentInstallState.Busy(appContext.getString(xendroid.compose.R.string.cm_deleting))
        val result = withContext(Dispatchers.IO) {
            runCatching {
                StorageAccess.acquire().use { lease ->
                    if (entry == null) trash.purgeAll(lease) else { trash.purge(lease, entry.id); 1 }
                }
            }
        }
        result.onSuccess { count ->
            _state.value = ContentInstallState.Done(if (entry == null) appContext.resources.getQuantityString(xendroid.compose.R.plurals.cm_emptied, count, count) else appContext.getString(xendroid.compose.R.string.cm_deleted, entry.displayName))
            refresh()
        }.onFailure { _state.value = ContentInstallState.Failed(it.message ?: appContext.getString(xendroid.compose.R.string.cm_delete_failed)) }
    }

    /** Skips the trash: the package's files are deleted now (asked when the trash is full). */
    fun deleteForGood(item: ContentEntry) = viewModelScope.launch {
        _deleteState.value = DeleteState.Idle
        _state.value = ContentInstallState.Busy(appContext.getString(xendroid.compose.R.string.cm_removing))
        val status = withContext(Dispatchers.IO) {
            val emu = EmulatorRuntime.emulator ?: return@withContext -1
            runCatching { StorageAccess.acquire().use { emu.delete_content(
                ContentPaths.contentRoot().absolutePath, titleOf(item),
                item.contentType, item.pkgDir) } }.getOrDefault(0xC0000022.toInt())
        }
        if (status == 0) {
            _state.value = ContentInstallState.Done(appContext.getString(xendroid.compose.R.string.cm_removed, item.displayName))
            refresh()
        } else {
            _state.value = ContentInstallState.Failed(deleteReasonFor(status))
        }
    }

    private sealed interface PreCheck {
        data class Ok(val name: String) : PreCheck
        data class Overwrite(val name: String) : PreCheck
        data class Reject(val message: String) : PreCheck
    }

    private suspend fun validate(srcPath: String): PreCheck {
        EmulatorRuntime.ensureLoaded()
        if (!File(srcPath).isFile) return PreCheck.Reject(appContext.getString(xendroid.compose.R.string.ci_open_failed))
        val meta = metadata.readContentHeader(srcPath)
            ?: return PreCheck.Reject(appContext.getString(xendroid.compose.R.string.ci_not_package))
        if (meta.contentType != ContentPaths.DLC_CONTENT_TYPE &&
            meta.contentType != ContentPaths.TU_CONTENT_TYPE)
            return PreCheck.Reject(
                appContext.getString(xendroid.compose.R.string.cm_wrong_type, meta.contentType.toUInt().toString(16)))
        // One game's screen installs that game's packages only; the Content area takes any title's.
        if (meta.titleId == null || (titleId != null && !meta.titleId.equals(titleId, ignoreCase = true)))
            return PreCheck.Reject(
                appContext.getString(xendroid.compose.R.string.cm_wrong_title, meta.titleId ?: "?", titleId ?: "?"))
        val name = meta.displayName.ifBlank { File(srcPath).name }
        storageShortfall(appContext, meta.contentSize)?.let { return PreCheck.Reject(it) }
        val pkgDir = File(ContentPaths.contentDir(meta.titleId, meta.contentType), meta.packageName)
        return if (pkgDir.exists()) PreCheck.Overwrite(name) else PreCheck.Ok(name)
    }

    private suspend fun runInstall(srcPath: String, name: String) {
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
            _state.value = ContentInstallState.Done(
                appContext.getString(xendroid.compose.R.string.cm_installed, name))
            refresh()
        } else {
            _state.value = ContentInstallState.Failed(installReasonFor(appContext, status))
        }
    }

    private fun deleteReasonFor(status: Int): String = when (status) {
        -1 -> appContext.getString(xendroid.compose.R.string.pf_no_emulator)
        0xC000000D.toInt() -> appContext.getString(xendroid.compose.R.string.cm_bad_name)            // X_STATUS_INVALID_PARAMETER
        0xC0000034.toInt() -> appContext.getString(xendroid.compose.R.string.cm_already_removed)     // X_STATUS_OBJECT_NAME_NOT_FOUND
        0xC0000022.toInt() -> appContext.getString(xendroid.compose.R.string.cm_delete_files_failed) // X_STATUS_ACCESS_DENIED
        else -> appContext.getString(xendroid.compose.R.string.cm_delete_failed_code, status.toUInt().toString(16))
    }
}
