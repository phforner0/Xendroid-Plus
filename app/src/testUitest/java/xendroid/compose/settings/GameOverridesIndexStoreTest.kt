package xendroid.compose.settings

import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import org.junit.runner.RunWith

/**
 * The index as the app builds it, through [ConfigStore], with the native TOML tables faked
 * (shots/FakeCore.kt). It used to ask the store for the file of Title ID 00000000 to find the
 * folder, which the store refuses: every read failed, so no screen named a game's own values.
 */
@RunWith(AndroidJUnit4::class)
class GameOverridesIndexStoreTest {
    @get:Rule val folder = TemporaryFolder()

    @Test fun readsTheGamesOwnValuesThroughTheStore() {
        val global = File(folder.root, "xenia-canary.config.toml").apply { writeText("[GPU]\nframerate_limit = 60\n") }
        val dir = File(folder.root, "config").apply { mkdirs() }
        File(dir, "4D5309C9.config.toml").writeText("[GPU]\nframerate_limit = 30\n")
        File(dir, "4D530919.config.toml").writeText("[Vulkan]\nvulkan_lib_path = \"/drivers/a/libvulkan_freedreno.so\"\n")
        // Only a key the app doesn't show: nothing of its own to name.
        File(dir, "545407F2.config.toml").writeText("[GPU]\nnot_a_setting = true\n")
        val store = ConfigStore(ApplicationProvider.getApplicationContext(), global)

        assertEquals(dir, store.perGameConfigDir())
        assertEquals(
            mapOf(
                "4D530919" to mapOf("Vulkan|vulkan_lib_path" to "/drivers/a/libvulkan_freedreno.so"),
                "4D5309C9" to mapOf("GPU|framerate_limit" to "30"),
            ),
            GameOverridesIndex(store).read(),
        )
    }
}
