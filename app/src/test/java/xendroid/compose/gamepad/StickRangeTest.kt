package xendroid.compose.gamepad

import kotlinx.serialization.json.Json
import org.junit.Assert.assertEquals
import org.junit.Test

class StickRangeTest {
    @Test fun defaultOuterDeadzonePreservesExistingBehavior() {
        assertEquals(0f, StickRange.scaleAxis(0.05f, innerDeadzone = 0.08f, outerDeadzone = 0f), 1e-5f)
        assertEquals(0.5f, StickRange.scaleAxis(0.5f, innerDeadzone = 0.08f, outerDeadzone = 0f), 1e-5f)
        assertEquals(-0.82f, StickRange.scaleAxis(-0.82f, innerDeadzone = 0.08f, outerDeadzone = 0f), 1e-5f)
        assertEquals(0f, StickRange.scaleAxis(Float.NaN, innerDeadzone = 0.08f, outerDeadzone = 0.20f), 1e-5f)
        assertEquals(0f, StickRange.scaleAxis(Float.POSITIVE_INFINITY, innerDeadzone = 0.08f, outerDeadzone = 0.20f), 1e-5f)
    }

    @Test fun outerDeadzoneScalesShortThrowSticksToFullDeflection() {
        // Odin 3 left stick maxing out at -0.82 Y reaches -1.00 cleanly with 20% outer deadzone (#141).
        assertEquals(-1.0f, StickRange.scaleAxis(-0.82f, innerDeadzone = 0.08f, outerDeadzone = 0.20f), 1e-5f)
        assertEquals(1.0f, StickRange.scaleAxis(0.80f, innerDeadzone = 0.08f, outerDeadzone = 0.20f), 1e-5f)
        assertEquals(-0.5f, StickRange.scaleAxis(-0.40f, innerDeadzone = 0.08f, outerDeadzone = 0.20f), 1e-5f)
        assertEquals(0f, StickRange.scaleAxis(-0.07f, innerDeadzone = 0.08f, outerDeadzone = 0.20f), 1e-5f)
    }

    @Test fun stepperAndSanitizerClampCleanly() {
        assertEquals(0f, StickRange.sanitizeOuterDeadzone(Float.NaN), 1e-5f)
        assertEquals(StickRange.MAX_OUTER_DEADZONE, StickRange.sanitizeOuterDeadzone(0.80f), 1e-5f)
        assertEquals(0f, StickRange.sanitizeOuterDeadzone(-0.2f), 1e-5f)
        assertEquals(0.05f, StickRange.stepOuterDeadzone(0f, 1), 1e-5f)
        assertEquals(0f, StickRange.stepOuterDeadzone(0f, -1), 1e-5f)
        assertEquals(0.30f, StickRange.stepOuterDeadzone(0.30f, 1), 1e-5f)
    }

    @Test fun slotInputRouterAppliesOuterDeadzonesIndependently() {
        val sent = mutableListOf<Triple<Int, Int, Int>>()
        val router = SlotInputRouter(deadzone = 0.08f) { slot, key, _, value ->
            if (slot == 1) sent.add(Triple(slot, key, value))
        }
        router.setStickOuterDeadzones(left = 0.20f, right = 0f)
        // Android Y up is negative; inverted inside motion() -> positive KC_LTHUMB_DOWN / negative KC_LTHUMB_UP:
        // Wait: invert=true -> raw = -(-0.82f) = +0.82f -> KC_LTHUMB_DOWN positive +32767.
        // Push left-stick UP in Android (-0.82f) -> raw = +0.82f, or push LEFT (-0.82f X, invert=false) -> -32768.
        router.motion(
            slot = 1,
            lx = -0.82f,
            ly = -0.82f,
            rx = 0.82f,
            ry = 0f,
            lt = 0f,
            rt = 0f,
            hatX = 0f,
            hatY = 0f,
        )
        assertEquals(-32768, sent.last { it.second == SlotInputRouter.KC_LTHUMB_LEFT }.third)
        assertEquals(32767, sent.last { it.second == SlotInputRouter.KC_LTHUMB_DOWN }.third)
        val rxSent = sent.last { it.second == SlotInputRouter.KC_RTHUMB_RIGHT }.third
        assertEquals((0.82f * 32767f).toInt(), rxSent)
    }

    @Test fun oldControlOptionsJsonWithoutOuterDeadzonesDecodesToZero() {
        val oldJson = """{"touchCamera":true,"gyroCamera":false,"gyroAim":"ALWAYS","gyroSensitivity":"NORMAL","unbufferedInput":true,"rumble":"MEDIUM"}"""
        val json = Json { ignoreUnknownKeys = true }
        val dto = json.decodeFromString(ControlOptionsDto.serializer(), oldJson)
        val resolved = dto.resolve(ControlOptions())
        assertEquals(0f, resolved.leftStickOuterDeadzone, 1e-5f)
        assertEquals(0f, resolved.rightStickOuterDeadzone, 1e-5f)
    }
}
