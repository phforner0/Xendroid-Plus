package xendroid.compose.settings

import java.io.File

/**
 * The games with values of their own (the per-game TOML files next to the global config), so a
 * global setting can say "3 games use another value" and list them, and the Settings summary and
 * the Drivers area can name the games that change something. Reads only the schema's keys.
 * [dir] is the folder of the games' files and [valuesOf] reads one game's values; the app reads
 * them through [ConfigStore], the tests without the native TOML parser.
 */
class GameOverridesIndex(
    private val dir: () -> File,
    private val valuesOf: (titleId: String) -> Map<String, String>,
) {
    constructor(store: ConfigStore) : this(store::perGameConfigDir, { id -> ownValues(store, id) })

    /** Title ID to its own values (key to raw), for every game file with at least one. A file
     *  that can't be read is left out; an Error (the native library not loaded) reaches the caller. */
    fun read(): Map<String, Map<String, String>> {
        val titles = dir().listFiles().orEmpty()
            .mapNotNull { FILE.matchEntire(it.name)?.groupValues?.get(1)?.uppercase() }
            .filter { it != "00000000" }.distinct().sorted()
        return titles.associateWith { id ->
            try { valuesOf(id) } catch (e: Exception) { emptyMap() }
        }.filterValues { it.isNotEmpty() }
    }

    companion object {
        // The native side reads 4D5309C9.config.toml; external storage ignores case.
        private val FILE = Regex("(?i)([0-9a-f]{8})\\.config\\.toml")

        private fun ownValues(store: ConfigStore, titleId: String): Map<String, String> {
            val handle = store.openGameConfig(titleId)
            try {
                return SettingsSchema.allSettings.mapNotNull { s -> handle.getString(s.section, s.name)?.let { s.key to it } }.toMap()
            } finally { handle.closeDiscard() }
        }

        /** Setting key to the Title IDs with their own value of it. */
        fun byKey(games: Map<String, Map<String, String>>): Map<String, List<String>> =
            games.flatMap { (title, values) -> values.keys.map { it to title } }
                .groupBy({ it.first }, { it.second })
                .mapValues { (_, titles) -> titles.sorted() }
    }
}
