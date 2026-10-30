/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2024 Xenia Canary. All rights reserved.                          *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/apu/audio_media_player.h"

#include <algorithm>
#include "xenia/apu/audio_driver.h"
#include "xenia/apu/audio_system.h"
#include "xenia/apu/xma_context.h"
#include "xenia/base/logging.h"
#include "xenia/kernel/guest_scheduler.h"

extern "C" {
#if XE_COMPILER_MSVC
#pragma warning(push)
#pragma warning(disable : 4101 4244 5033)
#endif
#include "third_party/FFmpeg/libavcodec/avcodec.h"
#include "third_party/FFmpeg/libavformat/avformat.h"
#include "third_party/FFmpeg/libavformat/avio.h"
#if XE_COMPILER_MSVC
#pragma warning(pop)
#endif
}  // extern "C"

DEFINE_bool(enable_xmp, true, "Enables Music Player playback.", "APU");

namespace xe {
namespace apu {

int32_t InitializeAndOpenAvCodec(std::span<uint8_t> song_data,
                                 AVFormatContext*& format_context,
                                 AVCodecContext*& av_context) {
  AVIOContext* io_ctx =
      avio_alloc_context(song_data.data(), (int)song_data.size(), 0, nullptr,
                         nullptr, nullptr, nullptr);

  format_context = avformat_alloc_context();
  format_context->pb = io_ctx;

  int ret = avformat_open_input(&format_context, nullptr, nullptr, nullptr);
  if (ret < 0) {
    return ret;
  }
  // Processing data
  ret = avformat_find_stream_info(format_context, nullptr);
  if (ret < 0) {
    return ret;
  }
  AVStream* stream = format_context->streams[0];

  // find & open codec
  AVCodecParameters* codec = stream->codecpar;
  auto decoder = avcodec_find_decoder(codec->codec_id);
  av_context = avcodec_alloc_context3(decoder);

  // Fill codec context with codec parameters
  ret = avcodec_parameters_to_context(av_context, codec);
  if (ret < 0) {
    return ret;
  }

  ret = avcodec_open2(av_context, decoder, NULL);
  return ret;
}

// The song's frame as interleaved stereo floats, whatever its channels: the
// media player's driver plays stereo, and one built for the song's own count
// read past its blocks (mono: twice the samples there were) or folded
// interleaved 5.1 as the guest's big endian kind. Mono goes to both sides;
// more channels keep front left and right, with the front centre at -3 dB.
void ConvertAudioFrame(AVFrame* frame, const AVChannelLayout& stream_layout,
                       std::vector<float>* framebuffer) {
  const AVChannelLayout& layout =
      frame->ch_layout.nb_channels > 0 ? frame->ch_layout : stream_layout;
  const int channels = std::max(layout.nb_channels, 1);
  const size_t frames = size_t(std::max(frame->nb_samples, 0));
  framebuffer->reserve(framebuffer->size() + frames * 2);

  // Planar and packed integer formats too: FFmpeg hands these back for WMA
  // and MP3, and leaving them unhandled left the frame buffer holding whatever
  // was there before, which the driver played as full-scale noise.
  float scale;
  switch (frame->format) {
    case AV_SAMPLE_FMT_FLT:
    case AV_SAMPLE_FMT_FLTP:
      scale = 1.0f;
      break;
    case AV_SAMPLE_FMT_S16:
    case AV_SAMPLE_FMT_S16P:
      scale = 1.0f / 32768.0f;
      break;
    case AV_SAMPLE_FMT_S32:
    case AV_SAMPLE_FMT_S32P:
      scale = 1.0f / 2147483648.0f;
      break;
    default:
      // Silence beats noise: without this the buffer keeps stale samples.
      XELOGW("XMP: unhandled sample format {}, substituting silence",
             int(frame->format));
      framebuffer->insert(framebuffer->end(), frames * 2, 0.0f);
      return;
  }
  const bool planar =
      av_sample_fmt_is_planar(static_cast<AVSampleFormat>(frame->format));
  const int format = frame->format;
  // extended_data: data[] holds only the first 8 planes.
  auto sample = [&](int channel, size_t index) -> float {
    const uint8_t* plane = frame->extended_data[planar ? channel : 0];
    const size_t at = planar ? index : index * channels + channel;
    switch (format) {
      case AV_SAMPLE_FMT_FLT:
      case AV_SAMPLE_FMT_FLTP:
        return reinterpret_cast<const float*>(plane)[at];
      case AV_SAMPLE_FMT_S16:
      case AV_SAMPLE_FMT_S16P:
        return float(reinterpret_cast<const int16_t*>(plane)[at]) * scale;
      default:
        return float(reinterpret_cast<const int32_t*>(plane)[at]) * scale;
    }
  };

  int left = 0, right = channels > 1 ? 1 : 0, center = -1;
  if (channels > 2) {
    const int fl =
        av_channel_layout_index_from_channel(&layout, AV_CHAN_FRONT_LEFT);
    const int fr =
        av_channel_layout_index_from_channel(&layout, AV_CHAN_FRONT_RIGHT);
    const int fc =
        av_channel_layout_index_from_channel(&layout, AV_CHAN_FRONT_CENTER);
    left = fl >= 0 ? fl : 0;
    right = fr >= 0 ? fr : 1;
    center = fc;
  }
  for (size_t i = 0; i < frames; ++i) {
    float l = sample(left, i);
    float r = sample(right, i);
    if (center >= 0) {
      const float c = 0.707106781f * sample(center, i);
      l += c;
      r += c;
    }
    framebuffer->push_back(l);
    framebuffer->push_back(r);
  }
}

ProcessAudioResult ProcessAudioLoop(AudioMediaPlayer* player,
                                    AudioDriver* driver, AVFormatContext* s,
                                    AVCodecContext* avctx, int streamIndex) {
  AVPacket* packet = av_packet_alloc();
  AVFrame* frame = av_frame_alloc();
  std::vector<float> frameBuffer;

  while (av_read_frame(s, packet) >= 0) {
    if (!player->IsSongLoaded()) {
      av_frame_free(&frame);
      av_packet_free(&packet);
      return ProcessAudioResult::ForcedFinish;
    }

    if (packet->stream_index == streamIndex) {
      int ret = avcodec_send_packet(avctx, packet);
      if (ret < 0) {
        XELOGE("Error sending packet for decoding: {:X}", ret);
        break;
      }

      while (ret >= 0) {
        if (!player->IsSongLoaded()) {
          break;
        }

        ret = avcodec_receive_frame(avctx, frame);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
          break;
        }
        if (ret < 0) {
          XELOGW("Error during decoding: {:X}", ret);
          break;
        }

        ConvertAudioFrame(frame, avctx->ch_layout, &frameBuffer);
        player->ProcessAudioBuffer(&frameBuffer);
      }
    }
    av_packet_unref(packet);
  }

  av_frame_free(&frame);
  av_packet_free(&packet);
  return ProcessAudioResult::Successful;
}

AudioMediaPlayer::AudioMediaPlayer(apu::AudioSystem* audio_system,
                                   kernel::KernelState* kernel_state)
    : audio_system_(audio_system),
      kernel_state_(kernel_state),
      active_playlist_(nullptr),
      active_song_(nullptr) {};

AudioMediaPlayer::~AudioMediaPlayer() {
  Stop();
  // Thread::reset() only closes the handle; must wait for the worker to exit.
  worker_running_ = false;
  resume_fence_.Signal();
  if (worker_thread_) {
    xe::threading::Wait(worker_thread_.get(), false);
    worker_thread_.reset();
  }
  DeleteDriver();
};

void AudioMediaPlayer::WorkerThreadMain() {
  while (worker_running_) {
    if (!IsPlaying()) {
      resume_fence_.Wait();
    }

    if (!active_playlist_) {
      xe::threading::Sleep(std::chrono::milliseconds(500));
      continue;
    }

    if (active_song_) {
      Play();
    }
  }
}

void AudioMediaPlayer::Setup() {
  if (!cvars::enable_xmp) {
    return;
  }

  // sample_buffer_ptr_ = kernel_state_->memory()->SystemHeapAlloc(
  //     xe::apu::AudioDriver::kFrameSamplesMax);

  worker_running_ = true;
  worker_thread_ = threading::Thread::Create({}, [&] { WorkerThreadMain(); });
  worker_thread_->set_name("Audio Media Player");
};

X_STATUS AudioMediaPlayer::Play(uint32_t playlist_handle, uint32_t song_handle,
                                bool force) {
  auto playlist_itr = playlists_.find(playlist_handle);
  if (playlist_itr == playlists_.cend()) {
    return X_STATUS_UNSUCCESSFUL;
  }

  active_playlist_ = playlist_itr->second.get();

  // I've set it to false (force) for PGR3
  if (!IsIdle()) {
    Stop(false, force);
  }

  if (!song_handle) {
    active_song_ = active_playlist_->songs.cbegin()->get();
    resume_fence_.Signal();
    return X_STATUS_SUCCESS;
  }

  auto song_itr = std::find_if(
      active_playlist_->songs.cbegin(), active_playlist_->songs.cend(),
      [song_handle](const std::unique_ptr<XmpApp::Song>& song) {
        return song->handle == song_handle;
      });

  if (song_itr == active_playlist_->songs.cend()) {
    return X_STATUS_UNSUCCESSFUL;
  }

  active_song_ = song_itr->get();
  resume_fence_.Signal();
  return X_STATUS_SUCCESS;
}

void AudioMediaPlayer::Play() {
  std::span<uint8_t> song_buffer = LoadSongToMemory();
  if (song_buffer.empty()) {
    return;
  }

  AVFormatContext* formatContext = nullptr;
  AVCodecContext* codecContext = nullptr;
  InitializeAndOpenAvCodec(song_buffer, formatContext, codecContext);

  // Stereo whatever the song's channels: ConvertAudioFrame folds them.
  if (!SetupDriver(codecContext->sample_rate, 2)) {
    XELOGE("Driver initialization failed!");
    avcodec_free_context(&codecContext);
    av_freep(&formatContext->pb->buffer);
    avio_context_free(&formatContext->pb);
    avformat_close_input(&formatContext);
    return;
  }

  state_ = XmpApp::State::kPlaying;
  current_song_handle_ = active_song_->handle;
  OnStateChanged();

  if (volume_ == 0.0f) {
    volume_ = kernel_state_->xconfig()->ReadSetting<float>(
        kernel::XCONFIG_USER_CATEGORY, kernel::XCONFIG_USER_MUSIC_VOLUME);
  }

  // Always apply the stored volume to the newly created driver
  driver_->SetVolume(volume_);

  auto result =
      ProcessAudioLoop(this, driver_.get(), formatContext, codecContext, 0);

  // We need to stop playback only if it wasn't
  if (result != ProcessAudioResult::ForcedFinish) {
    Stop(true, true);
  }

  // We're waiting for dangling samples to finish playing.
  if (result == ProcessAudioResult::Successful) {
    xe::threading::Wait(driver_semaphore_.get(), true,
                        std::chrono::milliseconds(500));
  }

  // Cleanup after work
  avcodec_free_context(&codecContext);
  av_freep(&formatContext->pb->buffer);
  avio_context_free(&formatContext->pb);
  avformat_close_input(&formatContext);

  processing_end_fence_.Signal();

  if (result == ProcessAudioResult::ForcedFinish) {
    DeleteDriver();
    return;
  }

  if (!IsLastSongInPlaylist()) {
    Next();
    return;
  }

  // We're after last song in playlist
  if (IsInRepeatMode()) {
    Play(active_playlist_->handle, 0, true);
  };
}

void AudioMediaPlayer::Pause() {
  if (!IsPlaying()) {
    return;
  }

  {
    // DeleteDriver frees the driver under this lock, on the player's thread.
    std::unique_lock<xe_mutex> guard(driver_mutex_);
    if (driver_) {
      driver_->Pause();
    }
  }
  state_ = XmpApp::State::kPaused;
  OnStateChanged();
};

void AudioMediaPlayer::Stop(bool change_state, bool force) {
  if (IsIdle()) {
    return;
  }

  if (IsPaused()) {
    std::unique_lock<xe_mutex> guard(driver_mutex_);
    if (driver_) {
      driver_->Resume();
    }
  }

  state_ = XmpApp::State::kIdle;
  active_song_ = nullptr;

  if (!force) {
    // Reachable from the guest XMP shim, so on a fiber this must park rather
    // than block: a plain fence wait would stall the whole dispatch CPU,
    // including the fibers that could satisfy it.
    kernel::GuestScheduler::WaitOnFence(processing_end_fence_);
  }

  if (change_state) {
    OnStateChanged();
  }
};

void AudioMediaPlayer::Continue() {
  if (!IsPaused()) {
    return;
  }

  state_ = XmpApp::State::kPlaying;
  resume_fence_.Signal();
  {
    std::unique_lock<xe_mutex> guard(driver_mutex_);
    if (driver_) {
      driver_->Resume();
    }
  }
  OnStateChanged();
}

X_STATUS AudioMediaPlayer::Next() {
  if (!active_playlist_) {
    return X_STATUS_UNSUCCESSFUL;
  }

  if (active_song_) {
    Stop(false, false);
  }

  auto itr = std::find_if(active_playlist_->songs.cbegin(),
                          active_playlist_->songs.cend(),
                          [this](const std::unique_ptr<XmpApp::Song>& song) {
                            return song->handle == current_song_handle_;
                          });

  // There is no song with such ID?
  if (itr == active_playlist_->songs.cend()) {
    return X_STATUS_UNSUCCESSFUL;
  }

  itr = std::next(itr);

  if (itr != active_playlist_->songs.cend()) {
    active_song_ = itr->get();
    resume_fence_.Signal();
    return X_STATUS_SUCCESS;
  }

  active_song_ = active_playlist_->songs.cbegin()->get();
  resume_fence_.Signal();
  return X_STATUS_SUCCESS;
}

X_STATUS AudioMediaPlayer::Previous() {
  if (!active_playlist_) {
    return X_STATUS_UNSUCCESSFUL;
  }

  if (active_song_) {
    Stop(false, false);
  }

  auto itr = std::find_if(active_playlist_->songs.cbegin(),
                          active_playlist_->songs.cend(),
                          [this](const std::unique_ptr<XmpApp::Song>& song) {
                            return song->handle == current_song_handle_;
                          });

  // We're at the first entry, we need to go to the end.
  if (itr == active_playlist_->songs.cbegin()) {
    active_song_ = active_playlist_->songs.crbegin()->get();
    resume_fence_.Signal();
    return X_STATUS_SUCCESS;
  }

  itr = std::prev(itr);
  active_song_ = itr->get();
  resume_fence_.Signal();
  return X_STATUS_SUCCESS;
}

std::span<uint8_t> AudioMediaPlayer::LoadSongToMemory() {
  if (!active_song_) {
    return {};
  }

  // Find file based on provided path?
  vfs::File* vfs_file;
  vfs::FileAction file_action;
  X_STATUS result = kernel_state_->file_system()->OpenFile(
      nullptr, xe::to_utf8(active_song_->file_path),
      vfs::FileDisposition::kOpen, vfs::FileAccess::kGenericRead, false, true,
      &vfs_file, &file_action);

  if (result) {
    return {};
  }

  const size_t file_size = vfs_file->entry()->size();
  uint8_t* data =
      file_size ? static_cast<uint8_t*>(av_malloc(file_size)) : nullptr;
  if (!data) {
    vfs_file->Destroy();
    return {};
  }
  std::span<uint8_t> buffer = {data, file_size};

  size_t bytes_read = 0;
  result = vfs_file->ReadSync(buffer, 0, &bytes_read);
  vfs_file->Destroy();
  if (result != X_ERROR_SUCCESS) {
    // Read failed. We need to manually release resources from av_malloc.
    av_free(data);
    return {};
  }

  return buffer;
}

void AudioMediaPlayer::AddPlaylist(uint32_t handle,
                                   std::unique_ptr<XmpApp::Playlist> playlist) {
  if (playlists_.count(handle) != 0) {
    return;
  }

  if (playback_mode_ == XmpApp::PlaybackMode::kShuffle) {
    auto rng = std::default_random_engine{};
    std::shuffle(playlist->songs.begin(), playlist->songs.end(), rng);
  }

  playlists_.insert({handle, std::move(playlist)});
}

void AudioMediaPlayer::RemovePlaylist(uint32_t handle) {
  if (playlists_.count(handle) == 0) {
    return;
  }

  // TODO: Check if currently played song is from that playlist and stop
  // playback.
  if (active_playlist_ && active_song_) {
    Stop();
  }

  playlists_.erase(handle);
}

X_STATUS AudioMediaPlayer::SetVolume(float volume) {
  volume_.store(std::min(volume, 1.0f));

  std::unique_lock<xe_mutex> guard(driver_mutex_);
  if (!driver_) {
    return X_STATUS_UNSUCCESSFUL;
  }

  driver_->SetVolume(volume_);
  return X_STATUS_SUCCESS;
}

bool AudioMediaPlayer::IsLastSongInPlaylist() const {
  if (!active_playlist_) {
    return false;
  }

  auto itr = std::find_if(active_playlist_->songs.cbegin(),
                          active_playlist_->songs.cend(),
                          [this](const std::unique_ptr<XmpApp::Song>& song) {
                            return song->handle == current_song_handle_;
                          });
  itr = std::next(itr);
  return itr == active_playlist_->songs.cend();
}

void AudioMediaPlayer::SetCaptureCallback(uint32_t callback, uint32_t context,
                                          bool title_render) {
  // TODO: Something is incorrect with callback and causes audio to be muted!
  callback_ = 0;
  callback_context_ = context;
  is_title_rendering_enabled_ = false;  // title_render;
}

void AudioMediaPlayer::OnStateChanged() {
  kernel_state_->BroadcastNotification(kXNotificationXmpStateChanged,
                                       static_cast<uint32_t>(state_));
}

void AudioMediaPlayer::ProcessAudioBuffer(std::vector<float>* buffer) {
  while (buffer->size() >= xe::apu::AudioDriver::kFrameSamplesMax) {
    xe::threading::Wait(driver_semaphore_.get(), true);

    if (!IsSongLoaded()) {
      buffer->clear();
      break;
    }

    // XMP should be processed on Audio thread. Some games goes insane when they
    // have additional unexpected guest thread (NG2 for example).
    /*
    if (callback_) {
      std::memcpy(kernel_state_->memory()->TranslateVirtual(sample_buffer_ptr_),
                  buffer->data(), xe::apu::AudioDriver::kFrameSamplesMax);

      uint64_t args[] = {sample_buffer_ptr_, callback_context_, true};
      audio_system_->processor()->Execute(worker_thread_->thread_state(),
                                          callback_, args, xe::countof(args));
    }*/

    driver_->SubmitFrame(buffer->data());
    buffer->erase(buffer->begin(),
                  buffer->begin() + xe::apu::AudioDriver::kFrameSamplesMax);
  }
}

bool AudioMediaPlayer::SetupDriver(uint32_t sample_rate, uint32_t channels) {
  DeleteDriver();

  std::unique_lock<xe_mutex> guard(driver_mutex_);
  driver_semaphore_ = xe::threading::Semaphore::Create(
      AudioSystem::kMaximumQueuedFrames, AudioSystem::kMaximumQueuedFrames);

  if (!driver_semaphore_) {
    return false;
  }

  driver_ = std::unique_ptr<AudioDriver>(audio_system_->CreateDriver(
      driver_semaphore_.get(), sample_rate, channels, false));

  if (!driver_) {
    driver_semaphore_.reset();
    return false;
  }

  if (!driver_->Initialize()) {
    // The driver before its semaphore, as in DeleteDriver.
    driver_->Shutdown();
    driver_.reset();
    driver_semaphore_.reset();
    return false;
  }

  return true;
}

void AudioMediaPlayer::DeleteDriver() {
  std::unique_lock<xe_mutex> guard(driver_mutex_);
  if (driver_) {
    // The driver first: its device callbacks (AAudio, OpenSL ES, on threads of
    // their own) release the semaphore until it is shut down. Freed first, the
    // semaphore could be released after it was gone.
    driver_->Shutdown();
    driver_.reset();
    driver_semaphore_.reset();
  }
}

}  // namespace apu
}  // namespace xe
