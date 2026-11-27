package xendroid.compose.driver

/**
 * 15p (Bannerlator `54c1dd35`): which listed driver suits this phone's GPU, read from the names
 * driver builders give their files and releases ("a7xx", "Adreno 740", "A8XX", "PanVK", "Xclipse").
 * Pure, tested on the JVM.
 */
object DriverSuggestion {
    /** How a driver's names relate to the GPU, best first. */
    enum class Fit { MODEL, FAMILY, ANY, OTHER_FAMILY }

    /** The Adreno model in a Vulkan device name ("Adreno (TM) 740", "Turnip Adreno (TM) 830"); null otherwise. */
    fun adrenoModel(gpu: String?): Int? =
        gpu?.let { Regex("(?i)adreno[^0-9]{0,8}(\\d{3,4})").find(it) }?.groupValues?.get(1)?.toIntOrNull()?.takeIf { it in 100..9999 }

    /** Mali / Immortalis model label ("Mali-G615", "Immortalis-G720"); null otherwise. */
    fun maliModel(gpu: String?): String? =
        gpu?.let { Regex("(?i)(immortalis-g\\d{2,4}|mali-g\\d{2,4})").find(it)?.groupValues?.get(1) }

    /** Samsung Xclipse model label ("Xclipse 940", "Xclipse 920"); null otherwise. */
    fun xclipseModel(gpu: String?): String? =
        gpu?.let { Regex("(?i)(xclipse\\s*\\d{3})").find(it)?.groupValues?.get(1)?.replace(Regex("\\s+"), " ") }

    /** Human-readable GPU chip label for the suggestion badge across Adreno, Mali and Xclipse. */
    fun gpuBadgeLabel(gpu: String?): String? =
        adrenoModel(gpu)?.toString() ?: maliModel(gpu) ?: xclipseModel(gpu)

    /** 6 for 6xx, 7 for 7xx, 8 for 8xx (an Adreno 702 is a 7xx). */
    fun family(model: Int): Int = if (model >= 1000) model / 1000 else model / 100

    private val FAMILY = Regex("(?i)(?<![a-z0-9])a?([5-9])xx(?![a-z0-9])")
    private val MODEL = Regex("(?i)(?:(?<![a-z0-9])a|adreno[ _-]?)(\\d{3})(?!\\d)")
    private val MALI_DRIVER = Regex("(?i)(malisx2|panvk|panfrost|mali|immortalis|valhall)")
    private val XCLIPSE_DRIVER = Regex("(?i)(xclipse|exynos)")
    private val ADRENO_ONLY_DRIVER = Regex("(?i)(armsx2|turnip|freedreno|adreno|qualcomm)")
    private val ARMSX2_DRIVER = Regex("(?i)armsx2")
    private val MALISX2_DRIVER = Regex("(?i)malisx2")

    fun isArmsx2Turnip(driver: DriverInfo): Boolean =
        ARMSX2_DRIVER.containsMatchIn("${driver.name} ${driver.version} ${driver.source}")

    fun isMalisx2(driver: DriverInfo): Boolean =
        MALISX2_DRIVER.containsMatchIn("${driver.name} ${driver.version} ${driver.source}")

    fun fit(driver: DriverInfo, model: Int): Fit {
        val text = driver.name + " " + driver.version
        if ((MALI_DRIVER.containsMatchIn(text) || isMalisx2(driver) || XCLIPSE_DRIVER.containsMatchIn(text)) &&
            !ADRENO_ONLY_DRIVER.containsMatchIn(text) && !isArmsx2Turnip(driver)
        ) {
            return Fit.OTHER_FAMILY
        }
        if (MODEL.findAll(text).any { it.groupValues[1].toInt() == model }) return Fit.MODEL
        val families = FAMILY.findAll(text).map { it.groupValues[1].toInt() }.toSet()
        return when {
            family(model) in families -> Fit.FAMILY
            families.isEmpty() && family(model) in 7..8 && isArmsx2Turnip(driver) -> Fit.FAMILY
            families.isEmpty() -> Fit.ANY
            else -> Fit.OTHER_FAMILY
        }
    }

    /** The suggestion for [gpu] among [drivers], or null when nothing fits this GPU vendor. */
    fun suggest(drivers: List<DriverInfo>, gpu: String?): DriverInfo? {
        val adreno = adrenoModel(gpu)
        if (adreno != null) {
            val preferArmsx2 = family(adreno) in 7..8
            return drivers.withIndex()
                .map { (index, driver) -> Triple(driver, fit(driver, adreno), index) }
                .filter { it.second != Fit.OTHER_FAMILY }
                .sortedWith(compareBy<Triple<DriverInfo, Fit, Int>> { it.second.ordinal }
                    .thenByDescending { it.first.publishedAt }
                    .thenByDescending { preferArmsx2 && isArmsx2Turnip(it.first) }
                    .thenBy { it.third })
                .firstOrNull()?.first
        }
        val mali = maliModel(gpu)
        if (mali != null) {
            return drivers.withIndex()
                .filter { (_, d) -> MALI_DRIVER.containsMatchIn("${d.name} ${d.version}") || isMalisx2(d) }
                .sortedWith(compareByDescending<IndexedValue<DriverInfo>> {
                    "${it.value.name} ${it.value.version}".contains(mali, ignoreCase = true)
                }.thenByDescending { it.value.publishedAt }.thenBy { it.index })
                .firstOrNull()?.value
        }
        val xclipse = xclipseModel(gpu)
        if (xclipse != null) {
            return drivers.withIndex()
                .filter { (_, d) -> XCLIPSE_DRIVER.containsMatchIn("${d.name} ${d.version}") }
                .sortedWith(compareByDescending<IndexedValue<DriverInfo>> {
                    "${it.value.name} ${it.value.version}".contains(xclipse, ignoreCase = true)
                }.thenByDescending { it.value.publishedAt }.thenBy { it.index })
                .firstOrNull()?.value
        }
        return null
    }
}
