package xendroid.compose.settings

import android.content.Context
import xendroid.compose.Application
import java.io.File

/** Sources the live global config file and the bundled default template baseline. */
class ConfigStore(private val appContext: Context, private val fileOverride: File? = null) {

    /** The live, editable config the emulator reads at boot. */
    fun globalConfigFile(): File = fileOverride ?: Application.get_global_config_file()

    /** <app_data_dir>/config, the folder of the games' own files. The app_data_dir is the
     *  parent of the global config file (Application.get_app_data_dir() is package-private;
     *  its public sibling get_global_config_file() lives in it). */
    fun perGameConfigDir(): File = File(globalConfigFile().parentFile, "config")

    /** <app_data_dir>/config/<TITLE_ID>.config.toml — exactly the path native
     *  config::LoadGameConfig reads (config.cc:309-311). */
    fun perGameConfigFile(titleId: String): File {
        require(titleId.matches(Regex("(?i)[0-9a-f]{8}")) && titleId != "00000000") {
            "Invalid game Title ID"
        }
        return File(perGameConfigDir(), "${titleId.uppercase()}.config.toml")
    }

    /** Read-only sparse snapshot. Missing means empty; malformed input is never erased. */
    fun openGameConfig(titleId: String): ConfigHandle {
        val file = perGameConfigFile(titleId)
        return ConfigHandle.openString(if (file.exists()) file.readText() else "")
    }

    /** Reparse the latest file under lock, patch just the edited keys and atomically replace. */
    fun editGameConfig(titleId: String, edit: (ConfigHandle) -> Unit) {
        editFile(perGameConfigFile(titleId), "", removeEmpty = true, edit = edit)
    }

    fun editLiveConfig(edit: (ConfigHandle) -> Unit) {
        editFile(globalConfigFile(), defaultTemplateText(), removeEmpty = false, edit = edit)
    }

    /** The core's own default changes applied to the live config and dated ([ConfigMigrations]).
     *  A missing file is left alone: it is seeded from the template, dated, later. */
    fun applyCoreDefaultUpdates(updates: List<ConfigMigrations.Update>) {
        val file = globalConfigFile()
        if (updates.isEmpty() || !file.exists()) return
        ConfigFileTransaction.update(file) { original ->
            original?.let { ConfigMigrations.migrate(it, updates) ?: it }
        }
    }

    private fun editFile(file: File, missing: String, removeEmpty: Boolean, edit: (ConfigHandle) -> Unit) {
        ConfigFileTransaction.update(file) { original ->
            val handle = ConfigHandle.openString(original ?: missing)
            try {
                edit(handle)
                if (removeEmpty && handle.isEmpty()) null else handle.closeString()
            } finally {
                handle.closeDiscard()
            }
        }
    }

    /** Open the live config READ-ONLY from a text snapshot (a string handle: free it
     *  with [ConfigHandle.closeDiscard]). Every write goes through [editLiveConfig],
     *  which re-reads the latest file under the cross-process lock and replaces it
     *  atomically. There is deliberately no "open the file for writing" path: the old
     *  one rewrote the file unlocked and replaced an unparseable file with the
     *  template, erasing every user setting. */
    fun openLiveSnapshot(): ConfigHandle {
        val file = globalConfigFile()
        ensureLiveFileExists(file)
        return ConfigHandle.openString(file.readText())
    }

    /** Read-only baseline parsed from the bundled asset template, for modified-from-default diffing. */
    fun openTemplateBaseline(): ConfigHandle =
        ConfigHandle.openString(defaultTemplateText())

    fun defaultTemplateText(): String = templateText
        ?: loadTemplateText().also { templateText = it }

    private var templateText: String? = null

    private fun loadTemplateText(): String =
        appContext.assets.open("config/default_config.toml").use { it.readBytes().toString(Charsets.UTF_8) }

    private fun ensureLiveFileExists(file: File) {
        if (file.exists()) return
        // Created under the same lock as every edit, so two processes racing to seed a
        // missing file cannot interleave a partial copy with an edit. An existing file
        // (even one that does not parse) is never replaced here.
        ConfigFileTransaction.update(file) { original ->
            original ?: Application.get_default_config_file().takeIf { it.isFile }?.readText()
                ?: defaultTemplateText()
        }
    }
}
