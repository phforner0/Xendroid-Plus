package xendroid.compose.settings

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

/** The redesign's groups and levels (docs/ui-redesign/bc): every setting placed once, sensibly. */
class SettingCatalogTest {
    @Test fun every_schema_setting_is_placed_exactly_once() {
        val placed = SettingCatalog.placedKeys
        assertEquals("placed twice: ${placed.groupBy { it }.filterValues { it.size > 1 }.keys}", placed.size, placed.toSet().size)
        assertEquals(SettingsSchema.allSettings.map { it.key }.toSet(), placed.toSet())
    }

    @Test fun essential_keeps_player_settings_plus_volume_vibration_and_zerofg() {
        val essential = SettingCatalog.settings(SettingLevel.ESSENTIAL).map { it.key }.toSet()
        assertEquals(SettingsSchema.playerKeys.toSet() + setOf("APU|volume", "HID|vibration", "Vulkan|zerofg_mode"), essential)
    }

    @Test fun levels_nest() {
        val e = SettingCatalog.settings(SettingLevel.ESSENTIAL).toSet()
        val a = SettingCatalog.settings(SettingLevel.ADVANCED).toSet()
        val x = SettingCatalog.settings(SettingLevel.ALL).toSet()
        assertTrue(a.containsAll(e))
        assertTrue(x.containsAll(a))
        assertEquals(SettingsSchema.allSettings.size, x.size)
    }

    @Test fun thirty_five_new_cvars_and_two_new_values() {
        assertEquals(35, SettingCatalog.newCount)
        assertEquals(2, SettingCatalog.newOptionCount)
        listOf("GPU|gpu_texture_lod_bias", "Vulkan|vulkan_android_present_divisor").forEach { key ->
            val meta = SettingCatalog.meta(key)
            assertEquals(SettingLevel.ADVANCED, meta.level)
            assertTrue("$key must be marked isNew", meta.isNew)
        }
        listOf(
            "GPU|vulkan_defer_resolve_clears",
            "GPU|vulkan_auto_vrs_thermal",
            "CPU|adpf_cpu_performance_hint",
            "CPU|jit_warmup_jprof",
        ).forEach { key ->
            val meta = SettingCatalog.meta(key)
            assertEquals(SettingGroup.PERFORMANCE, meta.group)
            assertEquals(SettingLevel.ADVANCED, meta.level)
            assertTrue("$key must be marked isNew", meta.isNew)
            assertFalse("$key must not be marked live", meta.live)
        }
        assertFalse("Kernel|kernel_delay0_backoff" in SettingCatalog.placedKeys)
        listOf(
            "CPU|dse_cross_block_context",
            "CPU|dse_cross_block_vec128",
            "CPU|a64_loop_context_promotion",
            "CPU|a64_loop_promote_max_regs",
            "CPU|a64_loop_promote_defer_stores_across_memory",
            "CPU|context_promote_vec128",
            "CPU|jit_warmup_jprof_threads",
            "CPU|jit_warmup_jprof_save_interval_ms",
            "CPU|jit_warmup_jprof_save_batch",
        ).forEach { key ->
            assertFalse("$key must stay TOML-only", key in SettingCatalog.placedKeys)
        }
    }

    @Test fun dependencies_name_real_settings_and_values() {
        for (key in SettingCatalog.placedKeys) {
            val dep = SettingCatalog.meta(key).dependsOn ?: continue
            val s = SettingsSchema.byKey[dep.key]
            assertNotNull("$key depends on a missing setting ${dep.key}", s)
            when (s) {
                is Setting.Bool -> assertTrue(dep.value == "true" || dep.value == "false")
                is Setting.ListChoice -> assertTrue("$key: ${dep.value} not an option of ${dep.key}", s.options.any { it.value == dep.value })
                else -> error("$key depends on ${dep.key}, which is not a switch or a list")
            }
        }
    }

    @Test fun a_dependent_setting_applies_only_with_the_needed_value() {
        val fsr = SettingsSchema.byKey["Display|postprocess_ffx_fsr_sharpness_reduction"]!!
        assertTrue(SettingCatalog.applies(fsr) { if (it == "Display|postprocess_scaling_and_sharpening") "fsr" else null })
        assertFalse(SettingCatalog.applies(fsr) { if (it == "Display|postprocess_scaling_and_sharpening") "cas" else null })
        val interp = SettingsSchema.byKey["GPU|async_shader_vs_interpreter"]!!
        assertTrue(SettingCatalog.applies(interp) { "true" })
        assertFalse(SettingCatalog.applies(interp) { "false" })
        // Unknown value of the other setting: shown as applying, never hidden by a guess.
        assertTrue(SettingCatalog.applies(interp) { null })
    }

    @Test fun live_settings_are_the_ones_the_in_game_menu_changes() {
        val live = SettingCatalog.placedKeys.filter { SettingCatalog.meta(it).live }.toSet()
        assertEquals(setOf("GPU|framerate_limit", "Display|postprocess_scaling_and_sharpening", "HID|show_touch_overlay", "APU|volume", "Vulkan|zerofg_mode"), live)
    }

    @Test fun pins_toggle() {
        val start = PinnedSettings.DEFAULT
        val without = PinnedSettings.toggled(start, "GPU|framerate_limit")
        assertFalse("GPU|framerate_limit" in without)
        assertEquals(start.toSet(), PinnedSettings.toggled(without, "GPU|framerate_limit").toSet())
    }
}
