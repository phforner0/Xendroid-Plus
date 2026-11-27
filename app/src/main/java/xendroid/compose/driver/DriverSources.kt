package xendroid.compose.driver

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.booleanOrNull

/**
 * 15p (Bannerlator `54c1dd35`, DroidDeck `f13c885`): where the driver manager looks for
 * drivers. Each source is a GitHub repository ("owner/repo") whose releases carry driver
 * packages as .zip assets; the manager starts with the usual one and the player may add others
 * (custom Turnip builds) or remove them. Only GitHub repositories, read through its API: what
 * a source points at is still checked like any download (the package's arm64 library, and its
 * SHA-256 when the release gives one). Pure, tested on the JVM.
 */
object DriverSources {
    const val DEFAULT = "K11MCH1/AdrenoToolsDrivers"
    const val ARMSX2_TURNIP = "bmdhacks/armsx2-turnip"
    const val MALISX2 = "bmdhacks/malisx2"
    const val MAX = 8

    /** Optional community repository offered for one-tap addition in the driver manager. */
    data class SuggestedSource(
        val repo: String,
        val label: String,
        val description: String,
    )

    val SUGGESTED: List<SuggestedSource> = listOf(
        SuggestedSource(
            repo = ARMSX2_TURNIP,
            label = "armsx2-turnip (Adreno A7xx/A8xx)",
            description = "Turnip builds with A7xx RB restore and opt-in KGSL/UBWC features",
        ),
        SuggestedSource(
            repo = MALISX2,
            label = "malisx2 (Mali PanVK-on-kbase)",
            description = "Mesa PanVK custom Vulkan driver builds for Arm Mali / Immortalis GPUs",
        ),
    )

    /** Suggested optional sources filtered/ordered for the detected GPU. */
    fun suggestedForGpu(gpu: String?): List<SuggestedSource> = when {
        DriverSuggestion.adrenoModel(gpu) != null -> SUGGESTED.filter { it.repo == ARMSX2_TURNIP }
        DriverSuggestion.maliModel(gpu) != null -> SUGGESTED.filter { it.repo == MALISX2 }
        else -> SUGGESTED
    }

    private val OWNER = Regex("[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})")
    private val REPO = Regex("[A-Za-z0-9._-]{1,100}")

    /**
     * "owner/repo" from what a person types or pastes: "owner/repo", "github.com/owner/repo",
     * "https://github.com/owner/repo/releases…" or "…/repo.git"; null when it is not one.
     */
    fun parse(input: String): String? {
        var text = input.trim()
        val scheme = Regex("^(?i)https?://").find(text)
        if (scheme != null) text = text.substring(scheme.range.last + 1)
        if (text.startsWith("www.", ignoreCase = true)) text = text.substring(4)
        val hosted = text.startsWith("github.com/", ignoreCase = true)
        if (hosted) text = text.substring("github.com/".length)
        else if (scheme != null) return null                     // a link to anywhere but GitHub
        val parts = text.split('/').filter { it.isNotEmpty() }
        // A bare "owner/repo" has nothing after it; a GitHub link may (/releases, /tree/…).
        if (parts.size < 2 || (!hosted && parts.size > 2)) return null
        val owner = parts[0]
        val repo = parts[1].removeSuffix(".git")
        if (!OWNER.matches(owner) || !REPO.matches(repo) || repo == "." || repo == "..") return null
        return "$owner/$repo"
    }

    fun releasesApi(source: String): String = "https://api.github.com/repos/$source/releases"

    /** The saved list (one per line); the usual source when nothing was saved. */
    fun decode(saved: String?): List<String> {
        if (saved == null) return listOf(DEFAULT)
        return saved.lines().mapNotNull { parse(it) }.distinctBy { it.lowercase() }.take(MAX)
    }

    fun encode(sources: List<String>): String = sources.joinToString("\n")

    enum class Added { ADDED, INVALID, ALREADY_THERE, FULL }

    /** [sources] with [input] added at the end, when it is a repository not yet listed. */
    fun add(sources: List<String>, input: String): Pair<List<String>, Added> {
        val source = parse(input) ?: return sources to Added.INVALID
        if (sources.any { it.equals(source, ignoreCase = true) }) return sources to Added.ALREADY_THERE
        if (sources.size >= MAX) return sources to Added.FULL
        return (sources + source) to Added.ADDED
    }

    fun remove(sources: List<String>, source: String): List<String> = sources.filterNot { it.equals(source, ignoreCase = true) }
}

/** A source's releases as the driver manager lists them (GitHub's releases API, newest first). */
object DriverReleases {
    private val json = Json { ignoreUnknownKeys = true }

    /**
     * Driver packages (.zip assets) of the published releases; drafts and pre-releases are left
     * out. An answer that is not a list of releases fails, so the source shows as unreadable.
     */
    fun parse(text: String, source: String): List<DriverInfo> {
        val releases = runCatching { json.parseToJsonElement(text) as? JsonArray }.getOrNull()
            ?: throw IllegalArgumentException("not a list of releases")
        return releases.mapNotNull { it as? JsonObject }.flatMap { release ->
            if (release.bool("draft") || release.bool("prerelease")) return@flatMap emptyList()
            val tag = release.string("tag_name")
            val name = release.string("name").ifBlank { tag }
            val published = release.string("published_at")
            (release["assets"] as? JsonArray).orEmpty().mapNotNull { it as? JsonObject }.mapNotNull { asset ->
                val assetName = asset.string("name")
                val url = asset.string("browser_download_url")
                if (!assetName.endsWith(".zip", ignoreCase = true) || url.isBlank()) return@mapNotNull null
                DriverInfo(
                    name = assetName,
                    version = "$name ($tag)",
                    url = url,
                    sha256 = githubAssetSha256(asset.string("digest").ifEmpty { null }).orEmpty(),
                    source = source,
                    publishedAt = published,
                )
            }
        }
    }

    private fun JsonObject.string(key: String): String = (this[key] as? JsonPrimitive)?.takeIf { it.isString }?.content.orEmpty()
    private fun JsonObject.bool(key: String): Boolean = (this[key] as? JsonPrimitive)?.booleanOrNull ?: false
}
