package xendroid.compose.driver

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class DriverSourcesTest {
    @Test fun whatPeopleTypeOrPasteBecomesOwnerSlashRepo() {
        assertEquals("K11MCH1/AdrenoToolsDrivers", DriverSources.parse("K11MCH1/AdrenoToolsDrivers"))
        assertEquals("Weab-chan/freedreno_turnip-CI", DriverSources.parse(" https://github.com/Weab-chan/freedreno_turnip-CI/releases "))
        assertEquals("owner/repo", DriverSources.parse("github.com/owner/repo.git"))
        assertEquals("owner/repo", DriverSources.parse("http://www.github.com/owner/repo/tree/main"))
        // Anything that is not a GitHub repository.
        assertNull(DriverSources.parse("https://gitlab.com/owner/repo"))
        assertNull(DriverSources.parse("https://example/owner/repo"))
        assertNull(DriverSources.parse("owner"))
        assertNull(DriverSources.parse("owner/repo/extra"))
        assertNull(DriverSources.parse("-owner/repo"))
        assertNull(DriverSources.parse("owner/.."))
        assertNull(DriverSources.parse("own er/repo"))
        assertNull(DriverSources.parse(""))
    }

    @Test fun theListIsKeptCleanAndStartsWithTheUsualSource() {
        assertEquals(listOf(DriverSources.DEFAULT), DriverSources.decode(null))
        assertEquals(emptyList<String>(), DriverSources.decode(""))                  // all removed stays removed
        assertEquals(listOf("a/b", "c/d"), DriverSources.decode("a/b\nnot a repo\nA/B\nc/d"))
        val many = (1..20).joinToString("\n") { "o/r$it" }
        assertEquals(DriverSources.MAX, DriverSources.decode(many).size)
        assertEquals(listOf("a/b", "c/d"), DriverSources.decode(DriverSources.encode(listOf("a/b", "c/d"))))
    }

    @Test fun addingAndRemoving() {
        val start = listOf(DriverSources.DEFAULT)
        val (added, ok) = DriverSources.add(start, "https://github.com/owner/turnip")
        assertEquals(DriverSources.Added.ADDED, ok)
        assertEquals(listOf(DriverSources.DEFAULT, "owner/turnip"), added)
        assertEquals(DriverSources.Added.ALREADY_THERE, DriverSources.add(added, "OWNER/turnip").second)
        assertEquals(DriverSources.Added.INVALID, DriverSources.add(added, "nope").second)
        val full = (1..DriverSources.MAX).map { "o/r$it" }
        assertEquals(DriverSources.Added.FULL, DriverSources.add(full, "x/y").second)
        assertEquals(listOf("owner/turnip"), DriverSources.remove(added, "k11mch1/adrenotoolsdrivers"))
        assertEquals("https://api.github.com/repos/owner/turnip/releases", DriverSources.releasesApi("owner/turnip"))
    }

    @Test fun releasesBecomeDriversWithTheirSourceAndDate() {
        val digest = "a".repeat(64)
        val json = """
            [
              {"name": "Turnip 25.1", "tag_name": "v25.1", "published_at": "2026-09-01T10:00:00Z", "draft": false,
               "prerelease": false, "assets": [
                 {"name": "turnip_a7xx.zip", "browser_download_url": "https://github.com/o/r/releases/download/v25.1/turnip_a7xx.zip",
                  "digest": "sha256:$digest"},
                 {"name": "notes.txt", "browser_download_url": "https://github.com/o/r/notes.txt"},
                 {"name": "broken.zip", "browser_download_url": ""}
               ]},
              {"name": "", "tag_name": "v25.2-rc", "prerelease": true, "assets": [
                 {"name": "rc.zip", "browser_download_url": "https://github.com/o/r/rc.zip"}]},
              {"tag_name": "v24", "draft": true, "assets": []},
              {"name": null, "tag_name": "v23", "assets": [
                 {"name": "old.zip", "browser_download_url": "https://github.com/o/r/old.zip", "digest": "md5:abc"}]}
            ]
        """.trimIndent()
        val drivers = DriverReleases.parse(json, "o/r")
        assertEquals(listOf("turnip_a7xx.zip", "old.zip"), drivers.map { it.name })
        assertEquals("Turnip 25.1 (v25.1)", drivers[0].version)
        assertEquals(digest, drivers[0].sha256)
        assertEquals("o/r", drivers[0].source)
        assertEquals("2026-09-01T10:00:00Z", drivers[0].publishedAt)
        assertEquals("v23 (v23)", drivers[1].version)
        assertEquals("", drivers[1].sha256)                 // a digest that is not SHA-256 is not checked
        // GitHub's answer for a missing repository is not a list: the source shows as unreadable.
        val failure = runCatching { DriverReleases.parse("""{"message": "Not Found"}""", "o/r") }
        assertTrue(failure.isFailure)
    }

    @Test fun theGpusFamilyPicksTheDriver() {
        assertEquals(740, DriverSuggestion.adrenoModel("Adreno (TM) 740"))
        assertEquals(830, DriverSuggestion.adrenoModel("Turnip Adreno (TM) 830"))
        assertNull(DriverSuggestion.adrenoModel("Mali-G715"))
        assertNull(DriverSuggestion.adrenoModel(null))
        assertEquals(7, DriverSuggestion.family(702))
        val generic = DriverInfo("Turnip_v25.0.0_R3.zip", "Turnip 25.0 (v25.0)", "u1", publishedAt = "2026-08-01")
        val a7 = DriverInfo("turnip_a7xx.zip", "Turnip 24.3 (v24.3)", "u2", publishedAt = "2026-06-01")
        val a8 = DriverInfo("Turnip-A8XX-25.1.zip", "Turnip 25.1 (v25.1)", "u3", publishedAt = "2026-09-01")
        val newerGeneric = DriverInfo("Turnip_v25.1.0_R1.zip", "Turnip 25.1 (v25.1)", "u4", publishedAt = "2026-09-02")
        val qualcomm = DriverInfo("Qualcomm_v762.zip", "Adreno 762 (v762)", "u5", publishedAt = "2026-09-03")
        val list = listOf(generic, a7, a8, newerGeneric)
        assertEquals(DriverSuggestion.Fit.FAMILY, DriverSuggestion.fit(a7, 740))
        assertEquals(DriverSuggestion.Fit.OTHER_FAMILY, DriverSuggestion.fit(a8, 740))
        assertEquals(DriverSuggestion.Fit.ANY, DriverSuggestion.fit(generic, 740))
        // A build named for the family beats a newer generic one; another family is never suggested.
        assertEquals(a7, DriverSuggestion.suggest(list, "Adreno (TM) 740"))
        assertEquals(a8, DriverSuggestion.suggest(list, "Adreno (TM) 830"))
        // No family build: the newest generic one.
        assertEquals(newerGeneric, DriverSuggestion.suggest(listOf(generic, newerGeneric, a8), "Adreno (TM) 650"))
        // A driver version that looks like a model ("v762") is not a model.
        assertEquals(DriverSuggestion.Fit.MODEL, DriverSuggestion.fit(qualcomm, 762))  // "Adreno 762" in its release name
        assertEquals(DriverSuggestion.Fit.ANY, DriverSuggestion.fit(qualcomm.copy(version = "v762"), 762))
        assertNull(DriverSuggestion.suggest(list, "Mali-G715"))
        assertNull(DriverSuggestion.suggest(listOf(a8), "Adreno (TM) 740"))
    }

    @Test fun optionalCommunitySourcesCoverArmsx2TurnipAndMalisx2() {
        assertEquals("K11MCH1/AdrenoToolsDrivers", DriverSources.DEFAULT)
        assertEquals(listOf(DriverSources.ARMSX2_TURNIP, DriverSources.MALISX2), DriverSources.SUGGESTED.map { it.repo })
        assertEquals(DriverSources.ARMSX2_TURNIP, DriverSources.parse(DriverSources.ARMSX2_TURNIP))
        assertEquals(DriverSources.MALISX2, DriverSources.parse(DriverSources.MALISX2))
        assertEquals(listOf(DriverSources.ARMSX2_TURNIP), DriverSources.suggestedForGpu("Adreno (TM) 825").map { it.repo })
        assertEquals(listOf(DriverSources.MALISX2), DriverSources.suggestedForGpu("Immortalis-G720").map { it.repo })
        assertEquals(2, DriverSources.suggestedForGpu(null).size)

        val genericTurnip = DriverInfo("Turnip_v25.2.0.zip", "Turnip 25.2 (v25.2)", "u1", publishedAt = "2026-10-10")
        val armsx2 = DriverInfo(
            "armsx2-turnip-25.2.zip",
            "armsx2-turnip (v25.2)",
            "u2",
            source = DriverSources.ARMSX2_TURNIP,
            publishedAt = "2026-10-05",
        )
        val malisx2 = DriverInfo(
            "malisx2-v25.2.zip",
            "malisx2 (v25.2)",
            "u3",
            source = DriverSources.MALISX2,
            publishedAt = "2026-10-08",
        )
        assertEquals(DriverSuggestion.Fit.FAMILY, DriverSuggestion.fit(armsx2, 740))
        assertEquals(DriverSuggestion.Fit.FAMILY, DriverSuggestion.fit(armsx2, 825))
        assertEquals(DriverSuggestion.Fit.ANY, DriverSuggestion.fit(armsx2, 650))
        assertEquals(DriverSuggestion.Fit.OTHER_FAMILY, DriverSuggestion.fit(malisx2, 825))
        assertEquals(armsx2, DriverSuggestion.suggest(listOf(genericTurnip, armsx2, malisx2), "Adreno (TM) 740"))
        assertEquals(armsx2, DriverSuggestion.suggest(listOf(genericTurnip, armsx2, malisx2), "Turnip Adreno (TM) 830"))
        assertEquals(malisx2, DriverSuggestion.suggest(listOf(genericTurnip, armsx2, malisx2), "Mali-G715"))
    }
}
