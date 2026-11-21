package xendroid.compose.gamepad

import kotlin.math.abs
import kotlin.math.roundToInt

/**
 * Stick axis scaling with inner dead zone and configurable outer dead zone (#141).
 * Hall-effect or handheld sticks that top out short of 1.0 (e.g. 0.82 on AYN Odin 3)
 * can set an outer dead zone so full physical throw maps cleanly to +/-1.0 in games.
 */
object StickRange {
    val OUTER_DEADZONE_STEPS: List<Float> = listOf(0f, 0.05f, 0.10f, 0.15f, 0.20f, 0.25f, 0.30f)
    const val MAX_OUTER_DEADZONE: Float = 0.30f

    fun sanitizeOuterDeadzone(value: Float): Float =
        if (!value.isFinite()) 0f else value.coerceIn(0f, MAX_OUTER_DEADZONE)

    fun scaleAxis(raw: Float, innerDeadzone: Float, outerDeadzone: Float = 0f): Float {
        if (!raw.isFinite()) return 0f
        val clamped = raw.coerceIn(-1f, 1f)
        if (abs(clamped) < innerDeadzone) return 0f
        val outer = sanitizeOuterDeadzone(outerDeadzone)
        if (outer <= 0f) return clamped
        val scale = (1f - outer).coerceAtLeast(0.5f)
        return (clamped / scale).coerceIn(-1f, 1f)
    }

    fun stepOuterDeadzone(current: Float, delta: Int): Float {
        val clean = sanitizeOuterDeadzone(current)
        val idx = OUTER_DEADZONE_STEPS.indices.minByOrNull { abs(OUTER_DEADZONE_STEPS[it] - clean) } ?: 0
        val next = (idx + delta).coerceIn(0, OUTER_DEADZONE_STEPS.lastIndex)
        return OUTER_DEADZONE_STEPS[next]
    }

    fun formatPercent(value: Float): String = "${(sanitizeOuterDeadzone(value) * 100f).roundToInt()}%"
}
