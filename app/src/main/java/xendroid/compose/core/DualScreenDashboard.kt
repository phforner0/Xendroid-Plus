package xendroid.compose.core

import android.app.Activity
import android.app.Presentation
import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.LinearGradient
import android.graphics.Paint
import android.graphics.Path
import android.graphics.RectF
import android.graphics.Shader
import android.graphics.Typeface
import android.hardware.display.DisplayManager
import android.os.Handler
import android.os.Looper
import android.view.Display
import android.view.View
import android.view.WindowManager
import java.util.Locale

/**
 * Clean-room secondary display telemetry & achievement dashboard built directly on the public
 * Android SDK (`android.app.Presentation` + `DisplayManager.DISPLAY_CATEGORY_PRESENTATION`).
 *
 * Automatically attaches to the secondary screen of dual-screen handhelds (AYN Thor, Pocket DS,
 * foldables) or external USB-C/HDMI monitors whenever [ExternalGameDisplay] is NOT routing the
 * primary Vulkan swapchain to that display. Renders via a zero-allocation custom [View] so it
 * never requires a `ViewTreeLifecycleOwner` inside `android.app.Presentation`.
 */
class DualScreenDashboard(
    private val activity: Activity,
    initialEnabled: Boolean = false,
) : DisplayManager.DisplayListener, AutoCloseable {

    data class DashboardSnapshot(
        val gameName: String = "XenDroid+",
        val titleId: String? = null,
        val driverLabel: String? = null,
        val fps: Double = 0.0,
        val frameMs: Double = 0.0,
        val playedSeconds: Long = 0L,
        val pipelinesCompiled: Long = 0L,
        val audioConcealed: Long = 0L,
        val batteryCelsius: Float = 0f,
        val controllerBatteryPercent: Int? = null,
        val recentEvents: List<String> = emptyList(),
    )

    private val manager = activity.getSystemService(DisplayManager::class.java)
    private var presentation: Presentation? = null
    private var canvasView: DashboardCanvasView? = null
    private var occupiedDisplayId: Int? = null
    private var enabled: Boolean = initialEnabled
    private var closing = false
    private var latest = DashboardSnapshot()
    private val fpsHistory = FloatArray(60)
    private var historyCount = 0

    init {
        manager?.registerDisplayListener(this, Handler(Looper.getMainLooper()))
        if (enabled) reconcileDisplay()
    }

    /** Enables or disables attaching telemetry to secondary/external displays (#146). */
    fun setEnabled(on: Boolean) {
        if (enabled == on) return
        enabled = on
        reconcileDisplay()
    }

    /** Informs the dashboard which display ID (if any) [ExternalGameDisplay] currently owns. */
    fun setGameOutputDisplay(displayId: Int?) {
        if (occupiedDisplayId == displayId) return
        occupiedDisplayId = displayId
        reconcileDisplay()
    }

    fun update(snapshot: DashboardSnapshot) {
        if (!enabled) return
        latest = snapshot
        if (snapshot.fps > 0.0) {
            if (historyCount < fpsHistory.size) {
                fpsHistory[historyCount++] = snapshot.fps.toFloat()
            } else {
                System.arraycopy(fpsHistory, 1, fpsHistory, 0, fpsHistory.size - 1)
                fpsHistory[fpsHistory.size - 1] = snapshot.fps.toFloat()
            }
        }
        if (presentation == null) reconcileDisplay()
        canvasView?.submit(latest, fpsHistory, historyCount)
    }

    private fun candidateDisplay(): Display? =
        manager?.getDisplays(DisplayManager.DISPLAY_CATEGORY_PRESENTATION)
            .orEmpty()
            .firstOrNull { it.isValid && it.displayId != Display.DEFAULT_DISPLAY && it.displayId != occupiedDisplayId }

    private fun reconcileDisplay() {
        if (closing || !enabled) {
            dismissCurrent()
            return
        }
        val wanted = candidateDisplay()
        val current = presentation?.display
        if (wanted == null) {
            dismissCurrent()
            return
        }
        if (current?.displayId == wanted.displayId && current.isValid) return
        dismissCurrent()
        try {
            val next = Presentation(activity, wanted)
            next.window?.addFlags(
                WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON or
                    WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE
            )
            val view = DashboardCanvasView(next.context).also {
                it.submit(latest, fpsHistory, historyCount)
            }
            next.setContentView(view)
            next.setOnDismissListener {
                if (!closing && presentation === next) {
                    presentation = null
                    canvasView = null
                }
            }
            next.show()
            presentation = next
            canvasView = view
        } catch (_: Throwable) {
            dismissCurrent()
        }
    }

    private fun dismissCurrent() {
        val old = presentation
        presentation = null
        canvasView = null
        runCatching { old?.dismiss() }
    }

    override fun onDisplayAdded(displayId: Int) = reconcileDisplay()
    override fun onDisplayRemoved(displayId: Int) {
        if (presentation?.display?.displayId == displayId) dismissCurrent()
        reconcileDisplay()
    }
    override fun onDisplayChanged(displayId: Int) = reconcileDisplay()

    override fun close() {
        closing = true
        runCatching { manager?.unregisterDisplayListener(this) }
        dismissCurrent()
    }

    private class DashboardCanvasView(context: Context) : View(context) {
        private var snap = DashboardSnapshot()
        private val samples = FloatArray(60)
        private var sampleCount = 0

        private val dp = resources.displayMetrics.density
        private val bgPaint = Paint(Paint.ANTI_ALIAS_FLAG)
        private val cardPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = 0xFF141C18.toInt() }
        private val borderPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = 1.2f * dp
            color = 0x337CE39B
        }
        private val titlePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.WHITE
            typeface = Typeface.create(Typeface.SANS_SERIF, Typeface.BOLD)
            textSize = 20f * dp
        }
        private val subPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = 0xFF9AB0A4.toInt()
            typeface = Typeface.MONOSPACE
            textSize = 12f * dp
        }
        private val kpiValuePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = 0xFF8BE89E.toInt()
            typeface = Typeface.create(Typeface.MONOSPACE, Typeface.BOLD)
            textSize = 22f * dp
        }
        private val kpiLabelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = 0xFF8CA095.toInt()
            typeface = Typeface.SANS_SERIF
            textSize = 11.5f * dp
        }
        private val graphStroke = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = 2.2f * dp
            color = 0xFF6BE38E.toInt()
        }
        private val eventPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = 0xFFD4E2DA.toInt()
            typeface = Typeface.MONOSPACE
            textSize = 12f * dp
        }
        private val rect = RectF()
        private val path = Path()

        fun submit(next: DashboardSnapshot, history: FloatArray, count: Int) {
            snap = next
            sampleCount = count.coerceIn(0, samples.size)
            System.arraycopy(history, 0, samples, 0, sampleCount)
            invalidate()
        }

        override fun onSizeChanged(w: Int, h: Int, oldw: Int, oldh: Int) {
            super.onSizeChanged(w, h, oldw, oldh)
            if (w > 0 && h > 0) {
                bgPaint.shader = LinearGradient(
                    0f, 0f, w.toFloat(), h.toFloat(),
                    0xFF0B110E.toInt(), 0xFF101B15.toInt(), Shader.TileMode.CLAMP
                )
            }
        }

        override fun onDraw(canvas: Canvas) {
            super.onDraw(canvas)
            val w = width.toFloat()
            val h = height.toFloat()
            if (w <= 0f || h <= 0f) return
            canvas.drawRect(0f, 0f, w, h, bgPaint)

            val pad = 18f * dp
            val radius = 14f * dp
            var y = pad

            // Header banner
            val headerH = 68f * dp
            rect.set(pad, y, w - pad, y + headerH)
            canvas.drawRoundRect(rect, radius, radius, cardPaint)
            canvas.drawRoundRect(rect, radius, radius, borderPaint)
            canvas.drawText(snap.gameName.take(42), pad + 16f * dp, y + 30f * dp, titlePaint)
            val meta = listOfNotNull(
                snap.titleId,
                snap.driverLabel?.take(34),
                formatDuration(snap.playedSeconds),
            ).joinToString(" · ")
            canvas.drawText(meta, pad + 16f * dp, y + 52f * dp, subPaint)
            y += headerH + 12f * dp

            // 3x2 KPI grid
            val cols = 3
            val gap = 10f * dp
            val tileW = (w - 2f * pad - (cols - 1) * gap) / cols
            val tileH = 66f * dp
            val kpis = arrayOf(
                "GUEST FPS" to String.format(Locale.US, "%.0f FPS", snap.fps),
                "FRAME TIME" to String.format(Locale.US, "%.1f ms", snap.frameMs),
                "AUDIO" to if (snap.audioConcealed == 0L) "Lock-Free OK" else "${snap.audioConcealed} xruns",
                "PIPELINES" to snap.pipelinesCompiled.toString(),
                "BATTERY TEMP" to if (snap.batteryCelsius > 0f) String.format(Locale.US, "%.1f C", snap.batteryCelsius) else "—",
                "PAD BATTERY" to (snap.controllerBatteryPercent?.let { "$it%" } ?: "—"),
            )
            for (i in kpis.indices) {
                val col = i % cols
                val row = i / cols
                val left = pad + col * (tileW + gap)
                val top = y + row * (tileH + gap)
                rect.set(left, top, left + tileW, top + tileH)
                canvas.drawRoundRect(rect, 12f * dp, 12f * dp, cardPaint)
                canvas.drawText(kpis[i].first, left + 12f * dp, top + 22f * dp, kpiLabelPaint)
                canvas.drawText(kpis[i].second, left + 12f * dp, top + 49f * dp, kpiValuePaint)
            }
            y += 2f * tileH + 2f * gap

            // Live 60s FPS waveform + Session feed
            val remainingH = (h - y - pad).coerceAtLeast(60f * dp)
            val graphH = (remainingH * 0.48f).coerceAtLeast(54f * dp)
            rect.set(pad, y, w - pad, y + graphH)
            canvas.drawRoundRect(rect, radius, radius, cardPaint)
            canvas.drawText("LIVE FPS HISTORY (60s)", pad + 14f * dp, y + 20f * dp, kpiLabelPaint)
            if (sampleCount >= 2) {
                path.reset()
                val gx0 = pad + 14f * dp
                val gx1 = w - pad - 14f * dp
                val gy0 = y + 28f * dp
                val gy1 = y + graphH - 10f * dp
                val maxFps = (samples.take(sampleCount).maxOrNull() ?: 60f).coerceAtLeast(60f)
                for (i in 0 until sampleCount) {
                    val x = gx0 + (gx1 - gx0) * (i.toFloat() / (sampleCount - 1).coerceAtLeast(1))
                    val norm = (samples[i] / maxFps).coerceIn(0f, 1f)
                    val py = gy1 - (gy1 - gy0) * norm
                    if (i == 0) path.moveTo(x, py) else path.lineTo(x, py)
                }
                canvas.drawPath(path, graphStroke)
            }
            y += graphH + gap

            val feedH = (h - y - pad).coerceAtLeast(0f)
            if (feedH >= 48f * dp) {
                rect.set(pad, y, w - pad, h - pad)
                canvas.drawRoundRect(rect, radius, radius, cardPaint)
                canvas.drawText("SESSION & ACHIEVEMENT FEED", pad + 14f * dp, y + 20f * dp, kpiLabelPaint)
                var ey = y + 40f * dp
                val events = snap.recentEvents.takeLast(4).reversed()
                if (events.isEmpty()) {
                    canvas.drawText("Dual-screen telemetry active", pad + 14f * dp, ey, eventPaint)
                } else {
                    for (line in events) {
                        if (ey > h - pad - 8f * dp) break
                        canvas.drawText(line.take(68), pad + 14f * dp, ey, eventPaint)
                        ey += 18f * dp
                    }
                }
            }
        }

        private fun formatDuration(seconds: Long): String {
            val s = seconds.coerceAtLeast(0L)
            val hrs = s / 3600
            val mins = (s % 3600) / 60
            val secs = s % 60
            return if (hrs > 0) String.format(Locale.US, "%d:%02d:%02d", hrs, mins, secs)
            else String.format(Locale.US, "%02d:%02d", mins, secs)
        }
    }
}
