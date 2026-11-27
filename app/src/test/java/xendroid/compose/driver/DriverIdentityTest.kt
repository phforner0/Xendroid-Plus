package xendroid.compose.driver

import org.junit.Assert.*
import org.junit.Test

class DriverIdentityTest {
    private val line = "vendor=0x5143;device=0x44050A00;driverVersion=0x80C00000;api=1.4.318;driverId=18;" +
        "driverName=turnip Mesa driver;driverInfo=Mesa 25.3.0-devel (git-1234abcd);gpu=Turnip Adreno (TM) 825;" +
        "uuid=0123456789ABCDEF0123456789abcdef"

    @Test fun parsesThePresenterLine() {
        val identity = DriverIdentity.parse(line)!!
        assertEquals("0x5143", identity.vendorId)
        assertEquals("0x44050A00", identity.deviceId)
        assertEquals(18, identity.driverId)
        assertEquals("0123456789abcdef0123456789abcdef", identity.uuid)
        assertEquals("turnip Mesa driver Mesa 25.3.0-devel (git-1234abcd) · Turnip Adreno (TM) 825", identity.label)
        assertEquals(12, identity.key.length)
    }

    @Test fun keySeparatesDriverBuildsButNotTheirNames() {
        val identity = DriverIdentity.parse(line)!!
        assertEquals(identity.key, identity.copy(driverName = "renamed", gpu = "other label").key)
        assertNotEquals(identity.key, identity.copy(uuid = "f".repeat(32)).key)
        assertNotEquals(identity.key, identity.copy(driverVersion = "0x80C00001").key)
    }

    @Test fun customLibraryIsTiedToItsInstalledPackage() {
        val sha = "ab".repeat(32)
        val custom = DriverIdentity.parse("$line;loader=custom;library=/data/user/0/xendroid.compose.fork/files/" +
            "custom_drivers/$sha/lib/libvulkan_freedreno.so")!!
        assertEquals("custom", custom.loader)
        assertEquals("libvulkan_freedreno.so", custom.library)   // the storage path is not kept
        assertEquals(sha, custom.packageSha256)
        assertTrue(custom.label.endsWith(" · package abababab"))
        // A reinstall beside a damaged copy is the same package.
        val reinstall = DriverIdentity.parse("$line;loader=custom;library=/x/$sha-0f8fad5b-d9cb-469f-a165-70867728950e/libvulkan.so")!!
        assertEquals(sha, reinstall.packageSha256)
        // Same driver build from two different packages: results are kept apart.
        assertNotEquals(custom.key, custom.copy(packageSha256 = "cd".repeat(32)).key)
        // The system driver keeps the key it had before packages were recorded.
        val system = DriverIdentity.parse("$line;loader=system;library=")!!
        assertEquals(DriverIdentity.parse(line)!!.key, system.key)
        assertNull(system.packageSha256)
        assertEquals("", system.library)
        // A library set by hand outside a package: named, but no package hash.
        val manual = DriverIdentity.parse("$line;loader=custom;library=/sdcard/turnip/libvulkan_freedreno.so")!!
        assertNull(manual.packageSha256)
        assertTrue(manual.label.endsWith(" · libvulkan_freedreno.so"))
        // A library path is ignored unless the presenter says a custom loader is in use.
        assertNull(DriverIdentity.parse("$line;loader=system;library=/x/$sha/libvulkan.so")!!.packageSha256)
    }

    @Test fun theInGameLineComparesWhatLoadedWithTheSettingThenAndNow() {
        val sha = "ab".repeat(32)
        val selected = "/data/x/$sha/libvulkan_freedreno.so"
        val custom = DriverIdentity.parse("$line;loader=custom;library=$selected")!!
        val system = DriverIdentity.parse("$line;loader=system;library=")!!
        assertEquals(DriverIdentity.InGame.UNKNOWN, DriverIdentity.inGame(selected, selected, null))
        assertEquals(DriverIdentity.InGame.AS_SELECTED, DriverIdentity.inGame(selected, selected, custom))
        assertEquals(DriverIdentity.InGame.AS_SELECTED, DriverIdentity.inGame("", "", system))
        // Custom chosen at start, system loaded: the fallback is said, whatever is selected now.
        assertEquals(DriverIdentity.InGame.CUSTOM_DID_NOT_LOAD, DriverIdentity.inGame(selected, "", system))
        // Changed while playing: applies at the next start.
        assertEquals(DriverIdentity.InGame.OTHER_FOR_NEXT_START, DriverIdentity.inGame("", selected, system))
        assertEquals(DriverIdentity.InGame.OTHER_FOR_NEXT_START, DriverIdentity.inGame(selected, "", custom))
        // Settings not read: only what loaded is shown.
        assertEquals(DriverIdentity.InGame.AS_SELECTED, DriverIdentity.inGame(null, null, custom))
    }

    @Test fun requestedDriverIsComparedWithWhatTheLastRunLoaded() {
        val sha = "ab".repeat(32)
        val custom = DriverIdentity.parse("$line;loader=custom;library=/data/x/$sha/libvulkan_freedreno.so")!!
        val system = DriverIdentity.parse("$line;loader=system;library=")!!
        val selected = "/data/x/$sha/libvulkan_freedreno.so"
        assertNull(DriverIdentity.describeEffective(selected, null, null, null))
        // The selection loaded.
        assertTrue(DriverIdentity.describeEffective(selected, custom, 2_000, 1_000)!!.startsWith("Last game ran on turnip"))
        assertFalse(DriverIdentity.describeEffective(selected, custom, 2_000, 1_000)!!.contains("did not load"))
        // A custom driver was selected but the system one loaded.
        assertTrue(DriverIdentity.describeEffective(selected, system, 2_000, 1_000)!!.endsWith("so the system driver was used"))
        // Another library than the selected one.
        assertTrue(DriverIdentity.describeEffective("/data/y/libvulkan_v37.so", custom, 2_000, 1_000)!!
            .endsWith("not the library selected now (libvulkan_v37.so)"))
        // The last run predates the selection: no verdict yet.
        assertTrue(DriverIdentity.describeEffective(selected, system, 500, 1_000)!!.startsWith("No game has run with this selection yet"))
        // System driver selected now, custom then.
        assertTrue(DriverIdentity.describeEffective("", custom, 2_000, null)!!.endsWith("(a custom driver was selected then)"))
        // Records from before the loader was known only say what ran.
        assertEquals("Last game ran on ${DriverIdentity.parse(line)!!.label}",
            DriverIdentity.describeEffective(selected, DriverIdentity.parse(line), 2_000, null))
    }

    @Test fun missingOrMalformedFieldsGiveNoIdentity() {
        assertNull(DriverIdentity.parse(""))
        assertNull(DriverIdentity.parse("vendor=0x5143;device=0x1"))                       // no uuid
        assertNull(DriverIdentity.parse(line.replace("uuid=0123456789ABCDEF", "uuid=xyz")))  // bad uuid
        assertNull(DriverIdentity.parse(line.replace("vendor=0x5143;", "")))
        assertNull(DriverIdentity.parse("x".repeat(5000)))
        // Unknown extra fields are ignored; absent optional ones default.
        val minimal = DriverIdentity.parse("vendor=0x1;device=0x2;uuid=${"0".repeat(32)};future=1")!!
        assertEquals("unknown driver", minimal.label)
        assertNull(minimal.driverId)
        assertEquals("driver 0x7", minimal.copy(driverVersion = "0x7").label)
    }

    @Test fun classifiesMalisx2PanvkArmsx2TurnipAndProprietaryFamiliesWithoutChangingKey() {
        val turnip = DriverIdentity.parse(line)!!
        assertFalse(turnip.isPanvk)
        assertFalse(turnip.isMalisx2)
        assertFalse(turnip.isArmsx2Turnip)
        assertEquals(DriverIdentity.DriverFamily.TURNIP, turnip.driverFamily)

        val armsx2 = turnip.copy(driverInfo = "Mesa 25.3.0-armsx2 (git-1234)")
        assertTrue(armsx2.isArmsx2Turnip)
        assertFalse(armsx2.isPanvk)
        assertEquals(DriverIdentity.DriverFamily.ARMSX2_TURNIP, armsx2.driverFamily)
        // Derived properties must never alter key stability.
        assertEquals(turnip.key, armsx2.key)

        val malisx2Id20 = DriverIdentity.parse(
            "vendor=0x13B5;device=0xB8A2;driverVersion=0x1903000;api=1.3.296;driverId=20;" +
                "driverName=panvk;driverInfo=Mesa 25.3.0 malisx2;gpu=Mali-G715;uuid=${"1".repeat(32)}",
        )!!
        assertTrue(malisx2Id20.isPanvk)
        assertTrue(malisx2Id20.isMalisx2)
        assertFalse(malisx2Id20.isArmsx2Turnip)
        assertEquals(DriverIdentity.DriverFamily.MALISX2, malisx2Id20.driverFamily)

        // PanVK-on-kbase wrapper reporting driverId=0 must still classify as MALISX2, not MALI_PROPRIETARY.
        val malisx2Id0 = malisx2Id20.copy(driverId = 0, driverName = "PanVK", driverInfo = "kbase wrapper")
        assertTrue(malisx2Id0.isPanvk)
        assertTrue(malisx2Id0.isMalisx2)
        assertEquals(DriverIdentity.DriverFamily.MALISX2, malisx2Id0.driverFamily)

        val maliProp = DriverIdentity.parse(
            "vendor=0x13B5;device=0xB8A2;driverVersion=0x2800000;api=1.3.264;driverId=9;" +
                "driverName=ARM;driverInfo=r40p0-00rel0;gpu=Mali-G715;uuid=${"2".repeat(32)}",
        )!!
        assertFalse(maliProp.isPanvk)
        assertFalse(maliProp.isMalisx2)
        assertEquals(DriverIdentity.DriverFamily.MALI_PROPRIETARY, maliProp.driverFamily)

        val qcomProp = DriverIdentity.parse(
            "vendor=0x5143;device=0x44050A00;driverVersion=0x802FA000;api=1.3.284;driverId=8;" +
                "driverName=Qualcomm;driverInfo=512.762.0;gpu=Adreno (TM) 740;uuid=${"3".repeat(32)}",
        )!!
        assertFalse(qcomProp.isArmsx2Turnip)
        assertEquals(DriverIdentity.DriverFamily.QUALCOMM_PROPRIETARY, qcomProp.driverFamily)
    }
}
