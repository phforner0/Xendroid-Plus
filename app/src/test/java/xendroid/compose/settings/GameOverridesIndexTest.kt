package xendroid.compose.settings

import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class GameOverridesIndexTest {
    @get:Rule val folder = TemporaryFolder()

    @Test fun listsTheGameFilesWithValuesOfTheirOwn() {
        val dir = folder.newFolder("config")
        for (name in listOf("4D5309C9.config.toml", "4D530919.config.toml", "545407F2.config.toml",
            "4d530aa4.config.toml", "00000000.config.toml", "4D530AA4.toml", "notes.txt")) File(dir, name).writeText("")
        val values = mapOf(
            "4D5309C9" to mapOf("GPU|framerate_limit" to "30"),
            "4D530919" to mapOf("Vulkan|vulkan_lib_path" to "/drivers/a/libvulkan_freedreno.so"),
            "4D530AA4" to mapOf("APU|apu_aaudio_buffer_bursts" to "8"),
        )
        // 545407F2 has a file but no value of its own, 00000000 is no game, the last two aren't
        // game files; a lowercase name is the same game (external storage ignores case).
        val read = GameOverridesIndex({ dir }) { id ->
            assertNotEquals("00000000", id)
            values[id].orEmpty()
        }.read()
        assertEquals(values, read)
        assertEquals(listOf("4D530919", "4D5309C9", "4D530AA4"), read.keys.toList())
    }

    @Test(expected = UnsatisfiedLinkError::class)
    fun aReaderWithoutItsLibraryFailsTheWholeRead() {
        // Not "no game has values": the view model logs it and keeps what it showed.
        val dir = folder.newFolder("config")
        File(dir, "4D5309C9.config.toml").writeText("")
        GameOverridesIndex({ dir }) { throw UnsatisfiedLinkError("libe.so not loaded") }.read()
    }

    @Test fun aGameThatCannotBeReadIsLeftOut() {
        val dir = folder.newFolder("config")
        for (id in listOf("4D5309C9", "4D530919")) File(dir, "$id.config.toml").writeText("")
        val read = GameOverridesIndex({ dir }) { id ->
            check(id != "4D5309C9") { "malformed" }
            mapOf("GPU|framerate_limit" to "30")
        }.read()
        assertEquals(mapOf("4D530919" to mapOf("GPU|framerate_limit" to "30")), read)
    }

    @Test fun noFolderMeansNoGames() {
        val read = GameOverridesIndex({ File(folder.root, "config") }) { error("no game to read") }.read()
        assertEquals(emptyMap<String, Map<String, String>>(), read)
    }
}
