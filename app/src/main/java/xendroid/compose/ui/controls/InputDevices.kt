package xendroid.compose.ui.controls

import android.content.Context
import android.content.SharedPreferences
import android.hardware.Sensor
import android.hardware.input.InputManager
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.os.VibrationEffect
import android.view.InputDevice
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.Stable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.runtime.staticCompositionLocalOf
import androidx.compose.ui.platform.LocalContext
import androidx.core.content.edit
import xendroid.compose.R
import xendroid.compose.gamepad.RumbleIntensity
import xendroid.compose.gamepad.RumbleSettings
import xendroid.compose.gamepad.TestedDevice
import xendroid.compose.gamepad.rumbleAmplitude
import xendroid.compose.ui.design.Gamepads

/** A keyboard the phone sees (not a controller): it plays through the key mapping. */
data class KeyboardDevice(val id: Int, val name: String)

/** The controllers and keyboards connected now. */
data class InputDevicesNow(val pads: List<TestedDevice> = emptyList(), val keyboards: List<KeyboardDevice> = emptyList())

/** Screenshots pass their devices here instead of the phone's. */
val LocalInputDevicesPreview = staticCompositionLocalOf<InputDevicesNow?> { null }

/** The phone's controllers and keyboards, described for the Controls area and the controller test. */
object InputDevices {
    fun describe(context: Context, device: InputDevice): TestedDevice {
        val sources = device.sources
        val names = listOfNotNull(
            context.getString(R.string.ct_src_gamepad).takeIf { sources and InputDevice.SOURCE_GAMEPAD == InputDevice.SOURCE_GAMEPAD },
            context.getString(R.string.ct_src_joystick).takeIf { sources and InputDevice.SOURCE_JOYSTICK == InputDevice.SOURCE_JOYSTICK },
            context.getString(R.string.ct_src_dpad).takeIf { sources and InputDevice.SOURCE_DPAD == InputDevice.SOURCE_DPAD },
            context.getString(R.string.ct_src_keyboard).takeIf { sources and InputDevice.SOURCE_KEYBOARD == InputDevice.SOURCE_KEYBOARD },
        )
        val vibrates = runCatching {
            if (Build.VERSION.SDK_INT >= 31) device.vibratorManager.vibratorIds.isNotEmpty()
            else @Suppress("DEPRECATION") device.vibrator.hasVibrator()
        }.getOrDefault(false)
        val gyro = Build.VERSION.SDK_INT >= 31 && runCatching { device.sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE) != null }.getOrDefault(false)
        return TestedDevice(
            device.id,
            device.name.orEmpty(),
            device.descriptor ?: "device:${device.id}",
            device.vendorId,
            device.productId,
            names,
            vibrates,
            gyro,
        )
    }

    /** A physical keyboard: letters, not the phone's own keys, not a controller. */
    fun isKeyboard(device: InputDevice?): Boolean {
        if (device == null || device.isVirtual || Gamepads.isController(device)) return false
        if (Build.VERSION.SDK_INT >= 29 && !device.isExternal) return false
        return device.keyboardType == InputDevice.KEYBOARD_TYPE_ALPHABETIC
    }

    fun now(context: Context): InputDevicesNow {
        val devices: List<InputDevice> = InputDevice.getDeviceIds().toList().mapNotNull { id -> InputDevice.getDevice(id) }
        return InputDevicesNow(
            pads = devices.filter { d -> Gamepads.isController(d) }.map { d -> describe(context, d) },
            keyboards = devices.filter { d -> isKeyboard(d) }.map { d -> KeyboardDevice(d.id, d.name.orEmpty()) },
        )
    }

    /** A short pulse at the controller's intensity, only because the player asked for it. */
    fun vibrate(id: Int, intensity: RumbleIntensity) {
        val device = InputDevice.getDevice(id) ?: return
        val amplitude = rumbleAmplitude(0xFFFFL, intensity)
        if (amplitude == 0) return
        val effect = VibrationEffect.createOneShot(300, amplitude)
        runCatching {
            if (Build.VERSION.SDK_INT >= 31) device.vibratorManager.defaultVibrator.vibrate(effect)
            else @Suppress("DEPRECATION") device.vibrator.vibrate(effect)
        }
    }
}

/** The connected controllers and keyboards; follows connections and disconnections. */
@Composable
fun rememberInputDevices(): InputDevicesNow {
    LocalInputDevicesPreview.current?.let { return it }
    val context = LocalContext.current
    var now by remember { mutableStateOf(runCatching { InputDevices.now(context) }.getOrDefault(InputDevicesNow())) }
    DisposableEffect(context) {
        val manager = context.getSystemService(Context.INPUT_SERVICE) as? InputManager
        val listener = object : InputManager.InputDeviceListener {
            private fun update() { now = runCatching { InputDevices.now(context) }.getOrDefault(InputDevicesNow()) }
            override fun onInputDeviceAdded(deviceId: Int) = update()
            override fun onInputDeviceRemoved(deviceId: Int) = update()
            override fun onInputDeviceChanged(deviceId: Int) = update()
        }
        manager?.registerInputDeviceListener(listener, Handler(Looper.getMainLooper()))
        onDispose { manager?.unregisterInputDeviceListener(listener) }
    }
    return now
}

/**
 * U08: each controller's own rumble intensity, kept by its Android descriptor in a file only
 * this (the app's) process writes; a game reads it when it starts. [default] is the shared one.
 */
@Stable
class PadRumbleState(private val prefs: SharedPreferences) {
    var settings by mutableStateOf(RumbleSettings.decode(null, prefs.getString(RumbleSettings.DEVICES_KEY, null)))
        private set

    fun own(descriptor: String): RumbleIntensity? = settings.perDevice[descriptor]

    /** [intensity] for this controller, or back to the default (null). */
    fun set(descriptor: String, intensity: RumbleIntensity?) {
        settings = settings.withDevice(descriptor, intensity)
        prefs.edit { putString(RumbleSettings.DEVICES_KEY, settings.encodeDevices()) }
    }
}

@Composable
fun rememberPadRumble(): PadRumbleState {
    val context = LocalContext.current
    return remember { PadRumbleState(context.getSharedPreferences(RumbleSettings.DEVICES_PREFS, Context.MODE_PRIVATE)) }
}
