package xendroid.compose.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Test

class SystemHeadroomReaderTest {
    class FakeSystemHealthManager(
        var cpu: Float = 50f,
        var gpu: Float = 60f,
        var cpuMinIntervalMs: Long = 1000L,
        var gpuMinIntervalMs: Long = 1000L,
        var throwOnCpu: Boolean = false,
        var throwOnGpu: Boolean = false,
    ) {
        var cpuCalls = 0
        var gpuCalls = 0

        @Suppress("UNUSED_PARAMETER")
        fun getCpuHeadroom(params: Any?): Float {
            cpuCalls++
            if (throwOnCpu) throw IllegalStateException("unsupported")
            return cpu
        }

        @Suppress("UNUSED_PARAMETER")
        fun getGpuHeadroom(params: Any?): Float {
            gpuCalls++
            if (throwOnGpu) throw IllegalStateException("unsupported")
            return gpu
        }

        fun getCpuHeadroomMinIntervalMillis(): Long = cpuMinIntervalMs
        fun getGpuHeadroomMinIntervalMillis(): Long = gpuMinIntervalMs
    }

    @Test fun returnsNullBelowApi36WithoutTouchingService() {
        var providerCalls = 0
        val reader = SystemHeadroomReader(
            sdkInt = 35,
            clockMs = { 1000L },
            serviceProvider = {
                providerCalls++
                FakeSystemHealthManager()
            },
        )
        assertNull(reader.read())
        assertEquals(0, providerCalls)
    }

    @Test fun readsCpuAndGpuHeadroomOnApi36AndFiltersInvalidValues() {
        var now = 2000L
        val fake = FakeSystemHealthManager(cpu = 42.5f, gpu = 14.0f, cpuMinIntervalMs = 500L, gpuMinIntervalMs = 500L)
        val reader = SystemHeadroomReader(sdkInt = 36, clockMs = { now }, serviceProvider = { fake })

        val first = reader.read()
        assertNotNull(first)
        assertEquals(42.5f, first!!.cpuHeadroom!!, 0f)
        assertEquals(14.0f, first.gpuHeadroom!!, 0f)

        // NaN and out-of-range values are filtered out cleanly.
        now = 4000L
        fake.cpu = Float.NaN
        fake.gpu = 105f
        assertNull(reader.read())

        now = 6000L
        fake.cpu = -1f
        fake.gpu = 0f
        val gpuOnly = reader.read()
        assertNotNull(gpuOnly)
        assertNull(gpuOnly!!.cpuHeadroom)
        assertEquals(0f, gpuOnly.gpuHeadroom!!, 0f)
    }

    @Test fun respectsMinPollingIntervalAndCachesWithinWindow() {
        var now = 5000L
        val fake = FakeSystemHealthManager(cpu = 80f, gpu = 70f, cpuMinIntervalMs = 2000L, gpuMinIntervalMs = 3000L)
        val reader = SystemHeadroomReader(sdkInt = 36, clockMs = { now }, serviceProvider = { fake })

        val s1 = reader.read()
        assertNotNull(s1)
        assertEquals(80f, s1!!.cpuHeadroom!!, 0f)
        assertEquals(70f, s1.gpuHeadroom!!, 0f)
        assertEquals(1, fake.cpuCalls)
        assertEquals(1, fake.gpuCalls)

        // 1s later: neither min interval elapsed -> returns cached values without calling service.
        now = 6000L
        fake.cpu = 30f
        fake.gpu = 20f
        val s2 = reader.read()
        assertNotNull(s2)
        assertEquals(80f, s2!!.cpuHeadroom!!, 0f)
        assertEquals(70f, s2.gpuHeadroom!!, 0f)
        assertEquals(1, fake.cpuCalls)
        assertEquals(1, fake.gpuCalls)

        // 2s after first read: CPU refreshes, GPU still cached.
        now = 7000L
        val s3 = reader.read()
        assertNotNull(s3)
        assertEquals(30f, s3!!.cpuHeadroom!!, 0f)
        assertEquals(70f, s3.gpuHeadroom!!, 0f)
        assertEquals(2, fake.cpuCalls)
        assertEquals(1, fake.gpuCalls)

        // 3s after first read: GPU refreshes too.
        now = 8000L
        val s4 = reader.read()
        assertNotNull(s4)
        assertEquals(30f, s4!!.cpuHeadroom!!, 0f)
        assertEquals(20f, s4.gpuHeadroom!!, 0f)
        assertEquals(2, fake.cpuCalls)
        assertEquals(2, fake.gpuCalls)
    }

    @Test fun degradesSilentlyWhenServiceMissingOrThrows() {
        val missing = SystemHeadroomReader(sdkInt = 36, clockMs = { 1000L }, serviceProvider = { null })
        assertNull(missing.read())

        val throwing = FakeSystemHealthManager(throwOnCpu = true, throwOnGpu = false, gpu = 25f)
        val reader = SystemHeadroomReader(sdkInt = 36, clockMs = { 1000L }, serviceProvider = { throwing })
        val partial = reader.read()
        assertNotNull(partial)
        assertNull(partial!!.cpuHeadroom)
        assertEquals(25f, partial.gpuHeadroom!!, 0f)
    }
}
