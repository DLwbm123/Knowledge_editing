"""Versioned LoRA-Perf/MedTRACE execution gates."""


def calibration_allowed(*, integration_pass: bool, effect_active: bool, v4_inputs_verified: bool) -> bool:
    return integration_pass and effect_active and v4_inputs_verified


def new_method_code_allowed(*, v4_inputs_verified: bool, method_spec_resolved: bool) -> bool:
    return v4_inputs_verified and method_spec_resolved


def new_method_gpu_allowed(*, lora_development_ready: bool, zero_effect_tests_pass: bool) -> bool:
    return lora_development_ready and zero_effect_tests_pass
