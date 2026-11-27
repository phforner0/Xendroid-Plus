package xendroid.compose.driver

import java.io.File
import java.nio.file.Files
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import xendroid.compose.driver.CustomDrivers.VendorFamily
import xendroid.compose.driver.DriverProbeActivity.ProbeOutcome

class CustomDriversAndProbeTest {

    @Test fun vendorDetectionCoversAdrenoMaliAndSamsungXclipse() {
        assertEquals(VendorFamily.ADRENO, CustomDrivers.detectVendor("Adreno (TM) 740"))
        assertEquals(VendorFamily.ADRENO, CustomDrivers.detectVendor("Turnip Adreno (TM) 830"))
        assertEquals(VendorFamily.ADRENO, CustomDrivers.detectVendor(null, hasKgsl = true))

        assertEquals(VendorFamily.MALI, CustomDrivers.detectVendor("Mali-G715 MC11"))
        assertEquals(VendorFamily.MALI, CustomDrivers.detectVendor("Immortalis-G720"))
        assertEquals(VendorFamily.MALI, CustomDrivers.detectVendor("PanVK Mali-G610"))
        assertEquals(VendorFamily.MALI, CustomDrivers.detectVendor(null, hasMaliNode = true))

        assertEquals(VendorFamily.XCLIPSE, CustomDrivers.detectVendor("Samsung Xclipse 940"))
        assertEquals(VendorFamily.XCLIPSE, CustomDrivers.detectVendor(null, socModel = "s5e9945"))
        assertEquals(VendorFamily.XCLIPSE, CustomDrivers.detectVendor(null, hasXclipseNode = true))

        assertEquals(VendorFamily.OTHER, CustomDrivers.detectVendor("PowerVR B-Series"))
        assertTrue(CustomDrivers.isSupportedVendor(VendorFamily.ADRENO))
        assertTrue(CustomDrivers.isSupportedVendor(VendorFamily.MALI))
        assertTrue(CustomDrivers.isSupportedVendor(VendorFamily.XCLIPSE))
        assertFalse(CustomDrivers.isSupportedVendor(VendorFamily.OTHER))
    }

    @Test fun driverSuggestionsSeparateTurnipPanVkAndXclipse() {
        val turnip7xx = DriverInfo("Turnip v25.3 a7xx", "25.3.0", "https://example/t7.zip", publishedAt = "2026-10-01")
        val panvk = DriverInfo("Mesa PanVK Valhall", "25.3.0", "https://example/panvk.zip", publishedAt = "2026-10-05")
        val xclipse940 = DriverInfo("Exynos Xclipse 940 Vulkan", "2.1.0", "https://example/xc940.zip", publishedAt = "2026-10-02")
        val catalog = listOf(panvk, xclipse940, turnip7xx)

        assertEquals("740", DriverSuggestion.gpuBadgeLabel("Adreno (TM) 740"))
        assertEquals("Mali-G715", DriverSuggestion.gpuBadgeLabel("Arm Mali-G715 MC11"))
        assertEquals("Immortalis-G720", DriverSuggestion.gpuBadgeLabel("Immortalis-G720 MC12"))
        assertEquals("Xclipse 940", DriverSuggestion.gpuBadgeLabel("Samsung Xclipse  940"))
        assertNull(DriverSuggestion.gpuBadgeLabel("PowerVR Rogue"))

        assertEquals(turnip7xx, DriverSuggestion.suggest(catalog, "Adreno (TM) 740"))
        assertEquals(panvk, DriverSuggestion.suggest(catalog, "Mali-G715"))
        assertEquals(xclipse940, DriverSuggestion.suggest(catalog, "Samsung Xclipse 940"))
        assertNull(DriverSuggestion.suggest(listOf(turnip7xx), "Mali-G715"))
        assertNull(DriverSuggestion.suggest(listOf(panvk), "Samsung Xclipse 940"))

        val malisx2 = DriverInfo(
            "malisx2_valhall.zip",
            "malisx2 25.3",
            "https://example/malisx2.zip",
            source = DriverSources.MALISX2,
            publishedAt = "2026-10-12",
        )
        val armsx2 = DriverInfo(
            "armsx2-turnip.zip",
            "armsx2 25.3",
            "https://example/armsx2.zip",
            source = DriverSources.ARMSX2_TURNIP,
            publishedAt = "2026-10-12",
        )
        val extendedCatalog = listOf(panvk, malisx2, xclipse940, turnip7xx, armsx2)
        assertEquals(malisx2, DriverSuggestion.suggest(extendedCatalog, "Immortalis-G720"))
        assertEquals(armsx2, DriverSuggestion.suggest(extendedCatalog, "Adreno (TM) 825"))
        assertEquals(xclipse940, DriverSuggestion.suggest(extendedCatalog, "Samsung Xclipse 940"))
        assertNull(DriverSuggestion.suggest(listOf(malisx2), "Adreno (TM) 825"))
        assertNull(DriverSuggestion.suggest(listOf(armsx2), "Mali-G715"))
    }

    @Test fun probeAllowsOnlyVerifiedDriverLibrariesAndParsesReplies() {
        val root = Files.createTempDirectory("drivers").toFile()
        val pkg = File(root, "pkg1").apply { mkdirs() }
        val validSo = File(pkg, "libvulkan_panvk.so").apply { writeText("elf") }
        val nonSo = File(pkg, "meta.json").apply { writeText("{}") }
        val outside = Files.createTempFile("outside", ".so").toFile()

        assertTrue(DriverProbeActivity.isAllowedDriverFile(validSo.absolutePath, root))
        assertFalse(DriverProbeActivity.isAllowedDriverFile(nonSo.absolutePath, root))
        assertFalse(DriverProbeActivity.isAllowedDriverFile(outside.absolutePath, root))
        assertFalse(DriverProbeActivity.isAllowedDriverFile(File(pkg, "../outside.so").path, root))
        assertFalse(DriverProbeActivity.isAllowedDriverFile(root.absolutePath, root))
        assertFalse(DriverProbeActivity.isAllowedDriverFile(validSo.absolutePath, null))

        assertEquals(
            ProbeOutcome.Passed("PanVK Mali-G715", "1.3.296"),
            DriverProbeActivity.parseNativeProbeReply("OK|PanVK Mali-G715|1.3.296"),
        )
        assertEquals(
            ProbeOutcome.Failed("vkCreateDevice returned -7"),
            DriverProbeActivity.parseNativeProbeReply("ERR|vkCreateDevice returned -7"),
        )
        assertTrue(DriverProbeActivity.parseNativeProbeReply("") is ProbeOutcome.Failed)
    }
}
