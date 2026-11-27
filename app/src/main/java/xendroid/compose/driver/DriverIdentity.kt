package xendroid.compose.driver

import java.security.MessageDigest
import kotlinx.serialization.Serializable

/**
 * The Vulkan driver build a guest actually ran on, as reported by the presenter when it
 * started — not the selected file name, which proves nothing about what loaded. The
 * pipelineCacheUUID decides whether a cached pipeline blob is usable; vendor/device and
 * driver version tell results of different drivers apart.
 */
@Serializable
data class DriverIdentity(
    val vendorId: String,
    val deviceId: String,
    val driverVersion: String,
    val api: String,
    val driverId: Int? = null,
    val driverName: String = "",
    val driverInfo: String = "",
    val gpu: String = "",
    val uuid: String,
    /** "custom" (a library loaded through adrenotools) or "system"; empty in older records. */
    val loader: String = "",
    /** File name of the custom library that loaded; empty for the system driver. */
    val library: String = "",
    /** SHA-256 of the installed package ([DriverPackageInstaller] directory) the library came from. */
    val packageSha256: String? = null,
) {
    /** Short stable key grouping results by driver build (and installed package, when there is one). */
    val key: String
        get() = MessageDigest.getInstance("SHA-256")
            .digest(("$vendorId|$deviceId|$driverVersion|$uuid" + (packageSha256?.let { "|$it" } ?: "")).toByteArray())
            .joinToString("") { "%02x".format(it) }.take(12)

    /** "Turnip Mesa 25.1.0 · Adreno (TM) 825 · package 1a2b3c4d", or the raw IDs when the driver gives no names. */
    val label: String
        get() {
            val driver = listOf(driverName, driverInfo).filter { it.isNotBlank() }.joinToString(" ")
                .ifBlank { if (driverVersion.isBlank()) "unknown driver" else "driver $driverVersion" }
            val source = when {
                packageSha256 != null -> " · package ${packageSha256.take(8)}"
                loader == "custom" && library.isNotBlank() -> " · $library"
                else -> ""
            }
            return (if (gpu.isBlank()) driver else "$driver · $gpu") + source
        }

    /** High-level Vulkan driver family derived from runtime properties (not serialized). */
    enum class DriverFamily {
        TURNIP,
        ARMSX2_TURNIP,
        PANVK,
        MALISX2,
        MALI_PROPRIETARY,
        QUALCOMM_PROPRIETARY,
        XCLIPSE_PROPRIETARY,
        RADV,
        OTHER,
    }

    /** True for open-source Mesa PanVK (including malisx2 PanVK-on-kbase wrappers with driverId 0). */
    val isPanvk: Boolean
        get() = driverId == 20 ||
            driverName.contains("panvk", ignoreCase = true) ||
            driverInfo.contains("malisx2", ignoreCase = true) ||
            driverName.contains("malisx2", ignoreCase = true)

    /** True for malisx2 (PanVK-on-kbase) builds on Arm Mali / Immortalis GPUs. */
    val isMalisx2: Boolean
        get() = isPanvk && (
            driverInfo.contains("malisx2", ignoreCase = true) ||
                driverName.contains("malisx2", ignoreCase = true) ||
                driverInfo.contains("kbase", ignoreCase = true) ||
                library.contains("malisx2", ignoreCase = true)
            )

    /** True for armsx2-turnip custom Turnip builds targeting Adreno A7xx/A8xx. */
    val isArmsx2Turnip: Boolean
        get() = (driverId == 18 || driverName.contains("turnip", ignoreCase = true)) && (
            driverInfo.contains("armsx2", ignoreCase = true) ||
                driverName.contains("armsx2", ignoreCase = true) ||
                library.contains("armsx2", ignoreCase = true)
            )

    val driverFamily: DriverFamily
        get() = when {
            isMalisx2 -> DriverFamily.MALISX2
            isPanvk -> DriverFamily.PANVK
            isArmsx2Turnip -> DriverFamily.ARMSX2_TURNIP
            driverId == 18 || driverName.contains("turnip", ignoreCase = true) -> DriverFamily.TURNIP
            driverId == 3 || driverName.contains("radv", ignoreCase = true) -> DriverFamily.RADV
            driverId == 9 || ((driverId == null || driverId == 0) && vendorId.equals("0x13B5", ignoreCase = true)) ->
                DriverFamily.MALI_PROPRIETARY
            driverId == 8 || ((driverId == null || driverId == 0) && vendorId.equals("0x5143", ignoreCase = true)) ->
                DriverFamily.QUALCOMM_PROPRIETARY
            driverId == 21 || ((driverId == null || driverId == 0) && vendorId.equals("0x144D", ignoreCase = true)) ->
                DriverFamily.XCLIPSE_PROPRIETARY
            else -> DriverFamily.OTHER
        }

    /** U01: what the in-game Graphics tab says about the driver of the running game. */
    enum class InGame {
        /** The presenter has not reported a driver yet. */
        UNKNOWN,
        /** The driver that loaded is the one the game started with. */
        AS_SELECTED,
        /** A custom driver was selected when the game started but the system one loaded. */
        CUSTOM_DID_NOT_LOAD,
        /** Another driver was selected while playing: it applies at the next start. */
        OTHER_FOR_NEXT_START,
    }

    companion object {
        /**
         * U01: [active] is what the presenter loaded; [requestedAtBoot] the driver setting
         * ("" = system) when the game started, [requestedNow] what it says now (null = not read).
         */
        fun inGame(requestedAtBoot: String?, requestedNow: String?, active: DriverIdentity?): InGame = when {
            active == null -> InGame.UNKNOWN
            !requestedAtBoot.isNullOrBlank() && active.loader == "system" -> InGame.CUSTOM_DID_NOT_LOAD
            requestedAtBoot != null && requestedNow != null && requestedNow != requestedAtBoot -> InGame.OTHER_FOR_NEXT_START
            else -> InGame.AS_SELECTED
        }

        /**
         * U03, requested vs effective: what the configured driver ([requestedPath], "" =
         * system) actually was in the latest run that recorded a driver. A run started
         * before the current selection ([selectedAt]) proves nothing about it. Null
         * without any recorded run.
         */
        fun describeEffective(requestedPath: String, last: DriverIdentity?, lastRunStartedAt: Long?, selectedAt: Long?): String? {
            last ?: return null
            val ran = "Last game ran on ${last.label}"
            return when (effective(requestedPath, last, lastRunStartedAt, selectedAt)) {
                Effective.NOT_RUN_YET -> "No game has run with this selection yet (the last one used ${last.label})"
                Effective.CUSTOM_DID_NOT_LOAD ->
                    "$ran: the selected custom driver did not load (missing or rejected), so the system driver was used"
                Effective.OTHER_LIBRARY -> "$ran, not the library selected now (${requestedPath.substringAfterLast('/')})"
                Effective.CUSTOM_THEN -> "$ran (a custom driver was selected then)"
                else -> ran
            }
        }

        /** What the latest run that recorded a driver says about the selection (for the screens' own words). */
        enum class Effective { NOT_RUN_YET, RAN, CUSTOM_DID_NOT_LOAD, OTHER_LIBRARY, CUSTOM_THEN }

        /** The case [describeEffective] words; null without any recorded run. */
        fun effective(requestedPath: String, last: DriverIdentity?, lastRunStartedAt: Long?, selectedAt: Long?): Effective? {
            last ?: return null
            if (selectedAt != null && lastRunStartedAt != null && lastRunStartedAt < selectedAt) return Effective.NOT_RUN_YET
            val requestedLibrary = requestedPath.substringAfterLast('/')
            return when {
                last.loader.isEmpty() -> Effective.RAN                    // recorded before the loader was known
                requestedPath.isNotBlank() && last.loader == "system" -> Effective.CUSTOM_DID_NOT_LOAD
                requestedPath.isNotBlank() && last.library.isNotBlank() && last.library != requestedLibrary -> Effective.OTHER_LIBRARY
                requestedPath.isBlank() && last.loader == "custom" -> Effective.CUSTOM_THEN
                else -> Effective.RAN
            }
        }

        private val uuidPattern = Regex("[0-9a-f]{32}")
        /** Content-addressed package directory: the archive SHA-256, or SHA-UUID for a reinstall beside a damaged copy. */
        private val packageDirectory = Regex("([0-9a-f]{64})(-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})?")

        /** Parses the presenter's "key=value;..." line; null when it is absent or malformed. */
        fun parse(text: String): DriverIdentity? {
            if (text.isBlank() || text.length > 2048) return null
            val fields = text.split(';').mapNotNull { part ->
                val pair = part.split('=', limit = 2)
                if (pair.size == 2) pair[0].trim() to pair[1].trim() else null
            }.toMap()
            val uuid = fields["uuid"]?.lowercase()?.takeIf { uuidPattern.matches(it) } ?: return null
            val loader = fields["loader"].orEmpty().take(16)
            // Only the file name and the package hash are kept, not the app's storage path.
            val segments = if (loader == "custom") fields["library"].orEmpty().split('/').filter { it.isNotEmpty() } else emptyList()
            val packageSha256 = segments.dropLast(1).firstNotNullOfOrNull {
                packageDirectory.matchEntire(it.lowercase())?.groupValues?.get(1)
            }
            return DriverIdentity(
                vendorId = fields["vendor"]?.takeIf { it.isNotBlank() } ?: return null,
                deviceId = fields["device"]?.takeIf { it.isNotBlank() } ?: return null,
                driverVersion = fields["driverVersion"].orEmpty().take(32),
                api = fields["api"].orEmpty().take(16),
                driverId = fields["driverId"]?.toIntOrNull(),
                driverName = fields["driverName"].orEmpty().take(64),
                driverInfo = fields["driverInfo"].orEmpty().take(128),
                gpu = fields["gpu"].orEmpty().take(128),
                uuid = uuid,
                loader = loader,
                library = segments.lastOrNull().orEmpty().take(128),
                packageSha256 = packageSha256,
            )
        }
    }
}
