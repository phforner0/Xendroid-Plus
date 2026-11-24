/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2020 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */
#ifndef XENDROID_XE_AAUDIO_AUDIO_DRIVER_H
#define XENDROID_XE_AAUDIO_AUDIO_DRIVER_H

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

#include <aaudio/AAudio.h>

#include "xenia/apu/audio_driver.h"
#include "xenia/base/cvar.h"
#include "xenia/base/threading.h"
#include "xe_audio_block_renderer.h"
#include "xe_audio_buffer_tuner.h"
#include "xe_audio_sanity.h"

DECLARE_uint32(apu_aaudio_buffer_bursts);
DECLARE_bool(apu_audio_sanity_log);

namespace xe {
namespace apu {
namespace aaudio {

class AAudioAudioDriver : public AudioDriver {
 public:
  // channels selects the submit contract, mirroring SDLAudioDriver: 6 is the
  // game path, 256 samples per channel of sequential big endian 5.1; 2 is the
  // media player, 768 frames of interleaved host endian stereo played at the
  // song's own rate. Both are 1536 floats per SubmitFrame.
  AAudioAudioDriver(Memory* memory, xe::threading::Semaphore* semaphore,
                    uint32_t frequency = 48000, uint32_t channels = 6,
                    bool need_format_conversion = true);
  ~AAudioAudioDriver() override;

  bool Initialize() override;
  void Pause() override;
  void Resume() override;
  void SetVolume(float volume) override;
  size_t GetQueuedFrameCount() override;
  bool ReportsQueuedFrameCount() const override { return true; }
  void SubmitFrame(float* frame) override;
  void Shutdown() override;

 protected:
  // Power-of-two capacity for the lock-free SPSC guest-block ringbuffer.
  // AudioSystem bounds in-flight submissions via semaphore_ (apu_max_queued_frames
  // <= 64), so 128 slots guarantees SubmitFrame and AudioCallback never block
  // or allocate on the real-time SCHED_FIFO callback thread.
  static constexpr uint32_t kRingCapacity = 128;
  static constexpr uint32_t kRingMask = kRingCapacity - 1;

  static aaudio_data_callback_result_t AudioCallback(
      AAudioStream* stream,
      void* userdata,
      void* audioData,
      int32_t numFrames);

  static void AudioErrorCallback(
      AAudioStream* stream,
      void* userdata,
      aaudio_result_t error);

  // (Re)opens the stream on the current default device; caller holds stream_mutex_.
  bool BuildStream();
  // Rebuilds the stream on the current default device; true once a stream is
  // live (or we're shutting down). Runs only on recovery_thread_.
  bool RestartStream();
  // Performs rebuild requests from the error callback, retrying on failure.
  void RecoveryThreadMain();
  // Drains queued blocks at real-time pace while no output stream is open
  // during Bluetooth / USB route transitions so guest audio-paced threads do
  // not freeze. Caller holds stream_mutex_ with stream_initialized_ == false.
  void DrainWhileDisconnectedLocked(uint32_t max_blocks);
  // 15j: one second's underruns into the adaptive buffer (apu_aaudio_adaptive_buffer).
  void AdaptBuffer(uint32_t fresh_xruns);

  xe::threading::Semaphore* semaphore_ = nullptr;

  AAudioStreamBuilder* builder_ = nullptr;
  AAudioStream* stream_ = nullptr;
  // Published with release on BuildStream / cleared before close so late error
  // callbacks from a disconnecting BT stream cannot tear down its replacement.
  std::atomic<AAudioStream*> active_stream_{nullptr};
  bool stream_initialized_ = false;
  // Serializes stream lifecycle. The data callback uses its `stream` argument,
  // not stream_, so it never takes this.
  std::mutex stream_mutex_ = {};

  // Output-device-change recovery: the error callback flags a request and
  // recovery_thread_ does the close+reopen (AAudio forbids closing from the
  // callback thread). Started in Initialize, joined in Shutdown.
  std::thread recovery_thread_ = {};
  std::mutex recovery_mutex_ = {};
  std::condition_variable recovery_cv_ = {};
  bool restart_requested_ = false;
  bool recovery_quit_ = false;
  std::atomic<bool> shutting_down_{false};

  static constexpr uint32_t host_frame_channels_ = 2;
  // Fixed by the creation contract, see the constructor.
  const uint32_t frame_frequency_;
  const uint32_t frame_channels_;
  const bool need_format_conversion_;
  const uint32_t channel_samples_;
  const uint32_t submit_samples_;
  const uint32_t host_block_samples_;

  // Pre-allocated contiguous SPSC ringbuffer storage + cacheline-separated
  // producer/consumer indices (zero mutexes, zero heap allocations in callback).
  std::vector<float> ring_storage_;
  alignas(64) std::atomic<uint32_t> ring_write_{0};
  alignas(64) std::atomic<uint32_t> ring_read_{0};

  // Conversion, gain, gap concealment and resampling of the guest blocks
  // (xe_audio_block_renderer.h). Callback thread only.
  AudioBlockRenderer renderer_;
  // A guest block has reached the device at least once (run summary counts).
  // Callback thread only.
  bool played_once_ = false;

  // Rate control state; callback thread only.
  float rate_ = 1.0f;
  // Set by Resume, taken by the next callback: the rate starts over at 1 there
  // (a write from Resume's thread could race a callback still running).
  std::atomic<bool> rate_reset_{false};

  // Loudness Guard + post-resume / post-250ms-stall ramp-in state
  // (ported cleanly from XDroid360 bf84217bb, BSD-3-Clause).
  std::atomic<bool> fade_request_{false};
  uint64_t last_callback_ns_ = 0;
  uint32_t post_stall_ramp_pos_ = 288;
  float limiter_gain_ = 1.0f;
  void ApplyLoudnessGuardAndRamp(float* stereo_out, int32_t num_frames,
                                 uint64_t now_ns);

  // The next queued guest block, or nullptr; and one given back once played.
  // Lock-free SPSC operations on ring_write_ / ring_read_.
  const float* NextGuestBlock();
  void ReturnGuestBlock(const float* block);

  // Written by the realtime callback, drained by recovery_thread_: relaxed
  // atomics only, nothing that could block the callback.
  std::atomic<uint64_t> stat_callbacks_{0};
  std::atomic<uint64_t> stat_gaps_{0};
  std::atomic<uint64_t> stat_queue_depth_sum_{0};
  std::atomic<uint32_t> stat_queue_depth_max_{0};
  // The shallowest the queue got: how much of it a stall of the producer used.
  std::atomic<uint32_t> stat_queue_depth_min_{UINT32_MAX};
  // A block size we did not ask for silently changes the drain rate.
  std::atomic<int32_t> stat_unexpected_frames_{0};
  // Samples the downmix pushed past full scale.
  std::atomic<uint64_t> stat_clipped_{0};
  std::atomic<uint32_t> stat_rate_milli_{1000};
  // The callbacks' own time (rendering), microseconds.
  std::atomic<uint64_t> stat_callback_us_sum_{0};
  std::atomic<uint32_t> stat_callback_us_max_{0};
  void LogAndResetStats();
  // Adds the stream's new xruns to ae::RunStats(); AAudio counts per stream, so
  // the count seen so far restarts with every rebuild. recovery_thread_ only.
  uint32_t PublishXRuns();
  int32_t xruns_published_ = 0;
  // 15j: the adaptive buffer's depth in bursts, and the frames one burst drains.
  AudioBufferTuner tuner_{cvars::apu_aaudio_buffer_bursts};
  int32_t drain_frames_ = 0;

  // Per-driver volume (XMP): written by other threads, read by the callback.
  std::atomic<float> driver_volume_{1.0f};

  // Callback-owned sanity tracker; published via relaxed atomics to recovery_thread_.
  xendroid::AudioSanityTracker audio_sanity_;
  uint32_t sanity_frames_accum_ = 0;
  std::atomic<uint32_t> sanity_rms_milli_{0};
  std::atomic<uint32_t> sanity_peak_milli_{0};
  std::atomic<uint64_t> sanity_silence_ms_{0};
  std::atomic<uint64_t> sanity_clipped_{0};
  std::atomic<uint64_t> sanity_nans_{0};
  std::atomic<bool> sanity_ready_{false};
};

}  // namespace aaudio
}  // namespace apu
}  // namespace xe

#endif  // XENDROID_XE_AAUDIO_AUDIO_DRIVER_H
