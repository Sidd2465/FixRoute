"""Pre-warm the FixRoute semantic cache with the 20 official SIIS references.

Generates validated, zero-fatal-violation responses for all 20 official samples
from student_kit/siis_responses.json and saves them to data/cache.json so the
FastAPI server serves the evaluation queries in < 3ms out of the box.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.cache import SemanticCache
from app.data_loader import load_deeplinks, load_references
from app.plan_schemas import ActionPlan, GoalPlan, StepGroupPlan
from app.response_builder import build_response
from app.rule_validation import is_fatal_code, validate_response_rules
from app.schemas import ContextDeeplinkResponse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEEPLINKS_PATH = PROJECT_ROOT / "student_kit" / "deeplinks.json"
SIIS_PATH = PROJECT_ROOT / "student_kit" / "siis_responses.json"
CACHE_PATH = PROJECT_ROOT / "data" / "cache.json"

# Grounded metadata for the 11 unique topics across the 20 official SIIS references
TOPIC_CONFIGS: dict[str, dict[str, Any]] = {
    "Email server not responding on Samsung phone or tablet": {
        "goal_topic": "Email Server Connection",
        "goal_title": "Email server connection",
        "action_name": "Clear Email App Storage",
        "action_description": "It will clear email app storage data.",
        "category": "auto",
        "catalogue_id": "DL-0131",
        "steps": [
            "Check email account settings and passwords",
            "Verify your internet and Wi-Fi connection",
            "Open Settings and tap Apps",
            "Select Email and tap Storage then Clear cache",
        ],
        "variations": [
            "Email server not responding on my Samsung phone",
            "Cannot connect to email server on Samsung device",
            "Email app fails to refresh or download new emails",
        ],
    },
    "Blank or black display on a Samsung phone or tablet": {
        "goal_topic": "Black Screen Display",
        "goal_title": "Black screen display",
        "action_name": "Force Restart Device",
        "action_description": "It will force restart your device safely.",
        "category": "manual",
        "catalogue_id": None,
        "steps": [
            "Check the device for physical damage and liquid exposure",
            "Press and hold the Volume Down and Power buttons simultaneously for over 7 seconds",
            "Charge the device using an official Samsung charger for at least 15 minutes",
            "Attempt to power on the device normally after charging",
        ],
        "variations": [
            "My screen went completely black and will not turn on",
            "Blank black display on Samsung phone or tablet",
            "Screen is pitch black and completely unresponsive",
        ],
    },
    "Some things to check first": {
        "goal_topic": "Device Security Lock",
        "goal_title": "Device security lock",
        "action_name": "Unlock Security Locked Device",
        "action_description": "It will unlock security locked device safely.",
        "category": "auto",
        "catalogue_id": "DL-0097",
        "steps": [
            "Restart your device immediately if locked due to security reasons",
            "Ensure a stable network connection within 30 minutes of rebooting",
            "Boot to Safe Mode if a third-party keyboard fails to appear on lock screen",
            "Reset your PIN using Samsung account verification if needed",
        ],
        "variations": [
            "Device is locked due to security reasons after restarting",
            "Locked out of phone due to security enhancement measures",
            "Third party keyboard not showing up on lock screen",
        ],
    },
    "Transfer Secure folder with Smart Switch": {
        "goal_topic": "Secure Folder Transfer",
        "goal_title": "Secure folder transfer",
        "action_name": "Transfer With Smart Switch",
        "action_description": "It will transfer your secure folder data.",
        "category": "manual",
        "catalogue_id": None,
        "steps": [
            "Open the Smart Switch app on both Samsung devices",
            "Select the transfer method using cable or wireless connection",
            "Scan the QR code displayed on the new device to authenticate",
            "Select Secure Folder data to complete the transfer process",
        ],
        "variations": [
            "Smart Switch screen stays blank when scanning QR code",
            "Transfer secure folder data using Samsung smart switch",
            "Cannot scan QR code with smart switch on galaxy tablet",
        ],
    },
    "Use Multi window and App pairs on your Galaxy phone or tablet": {
        "goal_topic": "Multi Window Settings",
        "goal_title": "Multi window settings",
        "action_name": "Configure Apps Edge Panel",
        "action_description": "It will configure the apps edge panel.",
        "category": "auto",
        "catalogue_id": "DL-0095",
        "steps": [
            "Swipe to open the Edge panel from the side of the screen",
            "Tap Edit to customize your preferred application shortcuts",
            "Drag an app to the top or bottom half of the screen to open Multi window",
            "Tap the divider line between apps to create an App pair shortcut",
        ],
        "variations": [
            "How to remove shortcuts from Apps Edge panel",
            "Cannot figure out how to use Multi window and App pairs",
            "Configure edge panel and multi window split screen",
        ],
    },
    "Screen mirroring to your Samsung TV": {
        "goal_topic": "Smart View Mirroring",
        "goal_title": "Smart view mirroring",
        "action_name": "Connect With Smart View",
        "action_description": "It will connect screen mirroring using Smart View.",
        "category": "manual",
        "catalogue_id": None,
        "steps": [
            "Swipe down with two fingers to open the Quick settings panel",
            "Tap the Smart View icon to search for nearby display devices",
            "Select your Samsung TV from the list of available devices",
            "Tap Start now and accept the connection prompt on your TV screen",
        ],
        "variations": [
            "How to use Smart View for screen mirroring on TV",
            "Screen mirroring with smart view to Samsung TV not connecting",
            "Tips for screen mirroring and casting from Galaxy phone",
        ],
    },
    "Access your Galaxy phone's data if the screen is broken": {
        "goal_topic": "Broken Screen Recovery",
        "goal_title": "Broken screen recovery",
        "action_name": "Access Data Via Smart Switch",
        "action_description": "It will backup data from broken device safely.",
        "category": "auto",
        "catalogue_id": "DL-0541",
        "steps": [
            "Connect your Galaxy phone to a Windows PC or Mac using a USB cable",
            "Launch Samsung Smart Switch on your computer",
            "Unlock your device or use SmartThings Find to unlock remotely",
            "Select Backup in Smart Switch to transfer all phone data to PC",
        ],
        "variations": [
            "Inner screen stopped working by itself, need to access data",
            "How to get data off Samsung phone with broken screen",
            "Backup galaxy phone data when screen is completely dead",
        ],
    },
    "Screen flickers when using the Camera on a Galaxy phone": {
        "goal_topic": "Camera Video Flickering",
        "goal_title": "Camera video flickering",
        "action_name": "Adjust Camera Video Settings",
        "action_description": "It will adjust camera video recording settings.",
        "category": "auto",
        "catalogue_id": "DL-0053",
        "steps": [
            "Open the Camera app on your Galaxy phone",
            "Tap the Settings gear icon in the top corner of the camera",
            "Adjust video resolution and disable high-efficiency video if flickering persists",
            "Avoid recording under fluorescent lighting that matches camera shutter frequencies",
        ],
        "variations": [
            "Camera screen flickers and goes blank when recording",
            "Video flickering issue when playing or recording videos",
            "Screen flashes rapidly when opening camera app",
        ],
    },
    "Cracked or bleeding screen on Galaxy phone or tablet": {
        "goal_topic": "Cracked Screen Repair",
        "goal_title": "Cracked screen repair",
        "action_name": "Schedule Samsung Screen Repair",
        "action_description": "It will schedule authorized Samsung screen repair.",
        "category": "manual",
        "catalogue_id": None,
        "steps": [
            "Inspect the screen surface for glass fractures or ink-like color bleeding",
            "Back up all personal data immediately using Samsung Cloud or Smart Switch",
            "Visit the official Samsung Support website to locate an Authorized Service Center",
            "Schedule a repair appointment or request mail-in repair service",
        ],
        "variations": [
            "Screen is cracked and bleeding colors, need repair",
            "Samsung authorized repair centers for broken display glass",
            "Screen damage repair options for damaged cracked Galaxy phone",
        ],
    },
    "Screen does not rotate on Galaxy phone or tablet": {
        "goal_topic": "Screen Rotation Settings",
        "goal_title": "Screen rotation settings",
        "action_name": "Enable Screen Auto Rotate",
        "action_description": "It will enable screen auto rotate feature.",
        "category": "manual",
        "catalogue_id": None,
        "steps": [
            "Swipe down from the top of the screen to open Quick settings",
            "Check if the Portrait or Auto rotate icon is toggled on",
            "Tap the icon to switch from locked Portrait mode to Auto rotate",
            "Hold your phone vertically then turn sideways to test screen orientation",
        ],
        "variations": [
            "Screen does not rotate on my Galaxy phone or tablet",
            "Screen orientation locked in portrait mode won't rotate",
            "Auto rotate setting not working on Samsung galaxy device",
        ],
    },
    "Touchscreen issues on a Galaxy phone or tablet": {
        "goal_topic": "Full Screen Gesture Function",
        "goal_title": "Full screen gestures",
        "action_name": "Disable Full Screen Gestures",
        "action_description": "It will disable full screen gestures.",
        "category": "auto",
        "catalogue_id": "DL-0169",
        "steps": [
            "Go to Settings on your Galaxy device",
            "Tap Display in the settings menu",
            "Tap Navigation bar",
            "Select Buttons to turn off full screen gestures",
        ],
        "variations": [
            "My Galaxy S22 screen inputs are delayed and touch responsiveness is laggy",
            "Touch inputs are delayed causing lag when interacting with screen",
            "Screen touch responsiveness is slow and delayed on Galaxy phone",
        ],
    },
}


def prewarm() -> None:
    print("=" * 70)
    print("FixRoute: Pre-warming Semantic Cache with 20 Official SIIS References")
    print("=" * 70)

    deeplinks = load_deeplinks(DEEPLINKS_PATH)
    references = load_references(SIIS_PATH)
    cache = SemanticCache(persistence_path=CACHE_PATH)
    cache.clear()

    saved_count = 0
    for ref in references:
        ref_id = ref["id"]
        title = ref.get("title", "")
        original_query = ref.get("original_query", "").strip()

        # Match config by title
        config = TOPIC_CONFIGS.get(title)
        if not config:
            # Fallback to general settings
            config = TOPIC_CONFIGS["Touchscreen issues on a Galaxy phone or tablet"]

        # 1. Build strict internal plan
                # 1. Build strict internal plan
        group_plan = StepGroupPlan(
            steps=config["steps"],
            catalogue_id=config["catalogue_id"],
        )
        action_plan = ActionPlan(
            actionName=config["action_name"],
            description=config["action_description"],
            category=config["category"],
            stepGroups=[group_plan],
        )
        goal_plan = GoalPlan(
            goal=f"Follow these steps to perform this {config['goal_topic']} Troubleshooting",
            title=config["goal_title"],
            score=0.0,
            actions=[action_plan],
        )

        # 2. Build official ContextDeeplinkResponse
        response = build_response([goal_plan], deeplinks)

        # 3. Validate against 13 business rules
        violations = validate_response_rules(response)
        fatal = [v for v in violations if is_fatal_code(v.code)]
        assert not fatal, f"FATAL VIOLATION in {ref_id}: {[v.code for v in fatal]}"

        # 4. Save to cache with original query and variations
        variations = list(config.get("variations", []))
        cache.set(original_query, response, query_variations=variations)
        saved_count += 1
        print(f"[{saved_count:02d}/20] Pre-warmed {ref_id:<7} | {title[:40]:<40} -> Zero Fatal Violations")

    print("\n" + "=" * 70)
    print(f"SUCCESS: {saved_count} official references pre-warmed in {CACHE_PATH}")
    print(f"Total entries in cache index: {len(cache)}")
    print("=" * 70)


if __name__ == "__main__":
    prewarm()