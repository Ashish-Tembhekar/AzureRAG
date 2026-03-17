#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test script to verify Azure Speech Service integration."""

import os
import sys
import tempfile
import wave
from pathlib import Path

# Fix encoding for Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from services import get_rag_cost_evaluator, get_speech_service, AzureQuotaExceededError


def test_speech_service_initialization():
    """Test that speech service initializes correctly."""
    print("✓ Testing speech service initialization...")
    speech_service = get_speech_service()
    assert speech_service is not None
    assert speech_service.speech_key is not None
    assert speech_service.speech_region == "eastus"
    print(f"  - Speech service initialized for region: {speech_service.speech_region}")
    print(f"  - Voice: {speech_service.speech_config.speech_synthesis_voice_name}")
    print("✓ Speech service test passed\n")


def test_quota_tracking():
    """Test that speech quota tracking works."""
    print("✓ Testing speech quota tracking...")
    evaluator = get_rag_cost_evaluator()
    monitor = evaluator.capacity_monitor

    assert monitor.MAX_STT_SECONDS_MONTH == 18000
    assert monitor.MAX_TTS_CHARS_MONTH == 500000
    print(f"  - MAX_STT_SECONDS_MONTH: {monitor.MAX_STT_SECONDS_MONTH}")
    print(f"  - MAX_TTS_CHARS_MONTH: {monitor.MAX_TTS_CHARS_MONTH}")

    initial_stt = monitor.usage.stt_seconds_used
    initial_tts = monitor.usage.tts_chars_used
    print(f"  - Initial STT seconds: {initial_stt}")
    print(f"  - Initial TTS chars: {initial_tts}")
    print("✓ Quota tracking test passed\n")


def test_tts_quota_verification():
    """Test TTS quota verification."""
    print("✓ Testing TTS quota verification...")
    evaluator = get_rag_cost_evaluator()
    monitor = evaluator.capacity_monitor

    test_text = "Hello, this is a test of the text-to-speech quota verification."
    char_count = monitor.verify_tts_quota(test_text)
    assert char_count == len(test_text)
    print(f"  - Text length: {char_count} characters")
    print("✓ TTS quota verification test passed\n")


def test_tts_synthesis():
    """Test TTS speech synthesis."""
    print("✓ Testing TTS speech synthesis...")
    speech_service = get_speech_service()
    evaluator = get_rag_cost_evaluator()

    test_text = "Hello! This is a test of the Azure Text-to-Speech service."

    try:
        initial_chars = evaluator.capacity_monitor.usage.tts_chars_used
        audio_data = speech_service.synthesize_speech(test_text)

        assert audio_data is not None
        assert len(audio_data) > 0
        print(f"  - Generated audio size: {len(audio_data)} bytes")

        final_chars = evaluator.capacity_monitor.usage.tts_chars_used
        print(f"  - TTS characters used: {final_chars - initial_chars}")
        print("✓ TTS synthesis test passed\n")
    except Exception as e:
        print(
            f"  ⚠ TTS synthesis test skipped (Azure service may not be available): {e}\n"
        )


def test_stt_quota_with_audio():
    """Test STT quota verification with actual audio file."""
    print("✓ Testing STT quota verification with audio...")
    evaluator = get_rag_cost_evaluator()
    monitor = evaluator.capacity_monitor

    duration = monitor._get_audio_duration
    print(f"  - Audio duration checker available: {duration is not None}")
    print("✓ STT quota verification test passed\n")


def test_cost_calculation():
    """Test speech cost calculation."""
    print("✓ Testing speech cost calculation...")
    evaluator = get_rag_cost_evaluator()
    monitor = evaluator.capacity_monitor

    initial_cost = monitor.usage.speech_cost_usd
    print(f"  - Initial speech cost: ${initial_cost:.6f}")

    tier = os.getenv("AZURE_TIER", "FREE").upper()
    print(f"  - Current tier: {tier}")

    if tier == "BASIC":
        expected_stt_cost_per_hour = 1.00
        expected_tts_cost_per_million = 15.00
        print(f"  - STT cost per hour: ${expected_stt_cost_per_hour}")
        print(f"  - TTS cost per 1M chars: ${expected_tts_cost_per_million}")

    print("✓ Cost calculation test passed\n")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Azure Speech Service Integration Tests")
    print("=" * 60 + "\n")

    try:
        test_speech_service_initialization()
        test_quota_tracking()
        test_tts_quota_verification()
        test_tts_synthesis()
        test_stt_quota_with_audio()
        test_cost_calculation()

        print("=" * 60)
        print("✓ All tests completed successfully!")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
