package xendroid.compose.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test
import xendroid.compose.core.ThermalWatch.Level

class ThermalWatchTest {
    @Test fun warnsBeforeTheDeviceThrottles() {
        val watch = ThermalWatch()
        assertNull(watch.sample(0.5f, 0))
        assertEquals(Level.NEAR_LIMIT, watch.sample(0.86f, 1))
        assertEquals(Level.THROTTLING, watch.sample(1.02f, 1))
        assertEquals(Level.NEAR_LIMIT, watch.sample(0.9f, 1))
        assertEquals(Level.OK, watch.sample(0.6f, 0))
    }

    @Test fun aReadingAroundTheLineDoesNotFlap() {
        val watch = ThermalWatch()
        watch.sample(0.86f, 0)
        assertNull(watch.sample(0.8f, 0))              // still above 0.7 once warned
        assertNull(watch.sample(0.72f, 0))
        assertEquals(Level.OK, watch.sample(0.69f, 0))
        assertNull(watch.sample(0.8f, 0))              // below 0.85 does not warn again
    }

    @Test fun theStatusAloneWhenThereIsNoHeadroom() {
        val watch = ThermalWatch()
        assertNull(watch.sample(null, 1))
        assertEquals(Level.NEAR_LIMIT, watch.sample(null, 2))
        assertEquals(Level.THROTTLING, watch.sample(Float.NaN, 3))
        assertEquals(Level.OK, watch.sample(null, 0))
        assertNull(watch.sample(-1f, -1))
    }

    @Test fun systemHeadroomNormalizesForJniWithoutAlteringAdvisoryLevel() {
        val watch = ThermalWatch()
        val (cpu1, gpu1) = watch.recordSystemHeadroom(45.5f, 12.0f)
        assertEquals(45.5f, cpu1, 0f)
        assertEquals(12.0f, gpu1, 0f)
        assertEquals(45.5f, watch.lastCpuHeadroom!!, 0f)
        assertEquals(12.0f, watch.lastGpuHeadroom!!, 0f)
        assertEquals(Level.OK, watch.level)

        val (cpuBad, gpuBad) = watch.recordSystemHeadroom(Float.NaN, 150f)
        assertEquals(-1f, cpuBad, 0f)
        assertEquals(-1f, gpuBad, 0f)
        assertNull(watch.lastCpuHeadroom)
        assertNull(watch.lastGpuHeadroom)
    }
}
