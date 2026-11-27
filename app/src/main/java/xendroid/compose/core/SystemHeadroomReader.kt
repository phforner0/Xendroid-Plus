package xendroid.compose.core

import android.content.Context
import android.os.Build
import android.os.SystemClock
import java.lang.reflect.Method

/**
 * Android 16 (API 36+) SystemHealthManager CPU/GPU headroom reader built against compileSdk 35
 * via cached reflection. Returns null immediately with zero overhead on API 29-35.
 *
 * Values follow Android's [0..100] inverted scale (0 = no compute/power headroom left, 100 = full
 * headroom). NaN and out-of-range readings are filtered to null; callers pass -1f to JNI when null.
 */
class SystemHeadroomReader(
    private val sdkInt: Int,
    private val clockMs: () -> Long,
    private val serviceProvider: () -> Any?,
) {
    data class Sample(val cpuHeadroom: Float?, val gpuHeadroom: Float?)

    private var initialized = false
    private var service: Any? = null
    private var getCpuHeadroomMethod: Method? = null
    private var getGpuHeadroomMethod: Method? = null
    private var cpuDefaultParams: Any? = null
    private var gpuDefaultParams: Any? = null
    private var cpuParamsResolved = false
    private var gpuParamsResolved = false
    private var cpuMinIntervalMs: Long = 0L
    private var gpuMinIntervalMs: Long = 0L

    private var lastCpuReadMs: Long = Long.MIN_VALUE
    private var lastGpuReadMs: Long = Long.MIN_VALUE
    private var lastCpuValue: Float? = null
    private var lastGpuValue: Float? = null

    fun read(): Sample? {
        if (sdkInt < MIN_API_LEVEL) return null
        ensureInitialized()
        val mgr = service ?: return null
        val nowMs = clockMs()

        val cpu = if (getCpuHeadroomMethod != null) {
            if (lastCpuReadMs == Long.MIN_VALUE || nowMs < lastCpuReadMs || (nowMs - lastCpuReadMs) >= cpuMinIntervalMs) {
                lastCpuReadMs = nowMs
                lastCpuValue = invokeHeadroom(mgr, getCpuHeadroomMethod!!, isCpu = true)
            }
            lastCpuValue
        } else null

        val gpu = if (getGpuHeadroomMethod != null) {
            if (lastGpuReadMs == Long.MIN_VALUE || nowMs < lastGpuReadMs || (nowMs - lastGpuReadMs) >= gpuMinIntervalMs) {
                lastGpuReadMs = nowMs
                lastGpuValue = invokeHeadroom(mgr, getGpuHeadroomMethod!!, isCpu = false)
            }
            lastGpuValue
        } else null

        if (cpu == null && gpu == null) return null
        return Sample(cpuHeadroom = cpu, gpuHeadroom = gpu)
    }

    private fun ensureInitialized() {
        if (initialized) return
        initialized = true
        val mgr = runCatching { serviceProvider() }.getOrNull() ?: return
        service = mgr
        val cls = mgr.javaClass
        getCpuHeadroomMethod = findHeadroomMethod(cls, "getCpuHeadroom")
        getGpuHeadroomMethod = findHeadroomMethod(cls, "getGpuHeadroom")
        cpuDefaultParams = resolveDefaultParamsOnce(getCpuHeadroomMethod)
        cpuParamsResolved = true
        gpuDefaultParams = resolveDefaultParamsOnce(getGpuHeadroomMethod)
        gpuParamsResolved = true
        cpuMinIntervalMs = invokeIntervalMethod(mgr, cls, "getCpuHeadroomMinIntervalMillis")
        gpuMinIntervalMs = invokeIntervalMethod(mgr, cls, "getGpuHeadroomMinIntervalMillis")
    }

    private fun resolveDefaultParamsOnce(method: Method?): Any? {
        if (method == null || method.parameterCount != 1) return null
        val paramType = method.parameterTypes[0]
        if (paramType == Any::class.java || paramType.isPrimitive) return null
        return buildDefaultParams(paramType)
    }

    private fun findHeadroomMethod(cls: Class<*>, name: String): Method? {
        val candidates = runCatching { cls.methods.filter { it.name == name } }.getOrDefault(emptyList())
        return candidates.firstOrNull { it.parameterCount == 1 }
            ?: candidates.firstOrNull { it.parameterCount == 0 }
    }

    private fun invokeIntervalMethod(mgr: Any, cls: Class<*>, name: String): Long {
        val method = runCatching { cls.getMethod(name) }.getOrNull() ?: return 0L
        val value = runCatching { (method.invoke(mgr) as? Number)?.toLong() }.getOrNull() ?: return 0L
        return value.coerceAtLeast(0L)
    }

    private fun invokeHeadroom(mgr: Any, method: Method, isCpu: Boolean): Float? {
        if (method.parameterCount == 0) {
            val raw = runCatching { (method.invoke(mgr) as? Number)?.toFloat() }.getOrNull()
            return sanitizeHeadroom(raw)
        }
        val params = if (isCpu) {
            if (!cpuParamsResolved) {
                cpuDefaultParams = resolveDefaultParamsOnce(method)
                cpuParamsResolved = true
            }
            cpuDefaultParams
        } else {
            if (!gpuParamsResolved) {
                gpuDefaultParams = resolveDefaultParamsOnce(method)
                gpuParamsResolved = true
            }
            gpuDefaultParams
        }
        if (params == null && method.parameterTypes[0].isPrimitive) {
            return null
        }
        val raw = runCatching { (method.invoke(mgr, params) as? Number)?.toFloat() }.getOrNull()
        return sanitizeHeadroom(raw)
    }

    private fun buildDefaultParams(paramClass: Class<*>): Any? = runCatching {
        val builderClass = Class.forName("${paramClass.name}\$Builder")
        val builder = builderClass.getDeclaredConstructor().newInstance()
        builderClass.getMethod("build").invoke(builder)
    }.getOrNull()

    companion object {
        const val MIN_API_LEVEL = 36
        const val UNAVAILABLE = -1f

        fun sanitizeHeadroom(raw: Float?): Float? =
            raw?.takeIf { it.isFinite() && it in 0f..100f }

        fun forContext(context: Context): SystemHeadroomReader = SystemHeadroomReader(
            sdkInt = Build.VERSION.SDK_INT,
            clockMs = SystemClock::elapsedRealtime,
            serviceProvider = { context.getSystemService("systemhealth") },
        )
    }
}
