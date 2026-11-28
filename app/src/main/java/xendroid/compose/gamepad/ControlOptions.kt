package xendroid.compose.gamepad

import android.content.Context
import android.content.SharedPreferences
import android.util.Log
import androidx.datastore.core.DataStore
import androidx.datastore.core.Serializer
import java.io.File
import java.io.InputStream
import java.io.OutputStream
import java.util.concurrent.ConcurrentHashMap
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.map
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

/** How far the phone's gyroscope turns the camera for the same motion (the in-game menu's choice). */
enum class GyroSensitivity(val scale: Float) {
    LOW(0.2f), NORMAL(0.35f), HIGH(0.6f);

    fun next(): GyroSensitivity = entries[(ordinal + 1) % entries.size]

    companion object {
        fun parse(name: String?): GyroSensitivity = entries.firstOrNull { it.name == name } ?: NORMAL
    }
}

/**
 * What every game starts with for input beyond the touch layout: the touch camera, the phone's
 * gyroscope, unbuffered input and the controllers' default rumble. The Controls area (app process)
 * changes them before a game, the in-game menu (game process) while one runs.
 */
data class ControlOptions(
    val touchCamera: Boolean = false,
    val gyroCamera: Boolean = false,
    val gyroAim: GyroAim = GyroAim.ALWAYS,
    val gyroSensitivity: GyroSensitivity = GyroSensitivity.NORMAL,
    val unbufferedInput: Boolean = true,
    val rumble: RumbleIntensity = RumbleIntensity.MEDIUM,
    val leftStickOuterDeadzone: Float = 0f,
    val rightStickOuterDeadzone: Float = 0f,
) {
    companion object {
        /** What older builds kept in the game process's own preferences (only ever read now). */
        fun legacy(prefs: SharedPreferences): ControlOptions = ControlOptions(
            touchCamera = prefs.getBoolean("touch_camera", false),
            gyroAim = GyroAim.parse(prefs.getString("gyro_aim", null)),
            unbufferedInput = prefs.getBoolean("unbuffered_input", true),
            rumble = RumbleIntensity.parse(prefs.getString(RumbleSettings.DEFAULT_KEY, null)),
        )
    }
}

/** The stored form: a value never set is null and comes from the older preferences. */
@Serializable
data class ControlOptionsDto(
    val touchCamera: Boolean? = null,
    val gyroCamera: Boolean? = null,
    val gyroAim: String? = null,
    val gyroSensitivity: String? = null,
    val unbufferedInput: Boolean? = null,
    val rumble: String? = null,
    val leftStickOuterDeadzone: Float? = null,
    val rightStickOuterDeadzone: Float? = null,
) {
    fun resolve(legacy: ControlOptions): ControlOptions = ControlOptions(
        touchCamera = touchCamera ?: legacy.touchCamera,
        gyroCamera = gyroCamera ?: legacy.gyroCamera,
        gyroAim = gyroAim?.let(GyroAim::parse) ?: legacy.gyroAim,
        gyroSensitivity = gyroSensitivity?.let(GyroSensitivity::parse) ?: legacy.gyroSensitivity,
        unbufferedInput = unbufferedInput ?: legacy.unbufferedInput,
        rumble = rumble?.let(RumbleIntensity::parse) ?: legacy.rumble,
        leftStickOuterDeadzone = StickRange.sanitizeOuterDeadzone(leftStickOuterDeadzone ?: legacy.leftStickOuterDeadzone),
        rightStickOuterDeadzone = StickRange.sanitizeOuterDeadzone(rightStickOuterDeadzone ?: legacy.rightStickOuterDeadzone),
    )

    companion object {
        fun of(o: ControlOptions) = ControlOptionsDto(
            o.touchCamera, o.gyroCamera, o.gyroAim.name, o.gyroSensitivity.name,
            o.unbufferedInput, o.rumble.name,
            StickRange.sanitizeOuterDeadzone(o.leftStickOuterDeadzone),
            StickRange.sanitizeOuterDeadzone(o.rightStickOuterDeadzone),
        )
    }
}

/**
 * [ControlOptions] in a file both processes write safely (a multi-process store, like the touch
 * layouts): preferences are cached per process and written whole, so two processes must never
 * share a preferences file.
 */
class ControlOptionsStore(private val appContext: Context) {
    private val store: DataStore<ControlOptionsDto> = stores.computeIfAbsent(appContext.filesDir.absolutePath) {
        SharedStores.create(ControlOptionsSerializer) { File(appContext.filesDir, "datastore/control-options.json") }
    }

    private fun legacy(): ControlOptions =
        ControlOptions.legacy(appContext.getSharedPreferences(RumbleSettings.PREFS, Context.MODE_PRIVATE))

    val options: Flow<ControlOptions> = store.data
        .catch {
            if (it is kotlinx.coroutines.CancellationException) throw it
            Log.w("ControlOptions", "Options unreadable; the defaults apply", it)
            emit(ControlOptionsDto())
        }
        .map { it.resolve(legacy()) }

    suspend fun update(transform: (ControlOptions) -> ControlOptions) {
        store.updateData { dto -> ControlOptionsDto.of(transform(dto.resolve(legacy()))) }
    }

    companion object { private val stores = ConcurrentHashMap<String, DataStore<ControlOptionsDto>>() }
}

object ControlOptionsSerializer : Serializer<ControlOptionsDto> {
    private val json = Json { ignoreUnknownKeys = true; encodeDefaults = true }
    override val defaultValue = ControlOptionsDto()
    override suspend fun readFrom(input: InputStream): ControlOptionsDto {
        val bytes = java.io.ByteArrayOutputStream()
        val buffer = ByteArray(4096)
        while (true) {
            val n = input.read(buffer)
            if (n < 0) break
            require(bytes.size() + n <= 64 * 1024) { "Control options are too large" }
            bytes.write(buffer, 0, n)
        }
        return if (bytes.size() == 0) ControlOptionsDto() else json.decodeFromString(ControlOptionsDto.serializer(), bytes.toString("UTF-8"))
    }
    override suspend fun writeTo(t: ControlOptionsDto, output: OutputStream) {
        output.write(json.encodeToString(ControlOptionsDto.serializer(), t).toByteArray())
    }
}
