import { useCallback, useRef } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { Invokes } from '../components/ui/AppProperties';
import { useEditorStore } from '../store/useEditorStore';
import { useProcessStore } from '../store/useProcessStore';

export interface DepthOfFieldState {
    enabled: boolean;
    focusDistance: number;  // 0.0 (near) to 1.0 (far) — which depth plane stays sharp
    blurAmount: number;     // Max blur radius in pixels, 0-40
    bokehThreshold: number; // Min depth offset for blur, 0.0-0.2 (focal range width)
    numRings: number;       // Concentric rings: 3-5 (quality vs speed tradeoff)
    samplesPerRing: number; // Samples per ring: 6-10
    bokehShape: 'circular' | 'hexagonal'; // Bokeh shape mode: 0=circular, 1=hexagonal
}

const DEFAULT_DOF: DepthOfFieldState = {
    enabled: false,
    focusDistance: 0.5,
    blurAmount: 8,
    bokehThreshold: 0.03,
    numRings: 3,
    samplesPerRing: 7,
    bokehShape: 'circular',
};

// Module-level state (simple approach; could be a Zustand store later)
let _state: DepthOfFieldState = { ...DEFAULT_DOF };

interface DepthOfFieldResult {
    preview_url: string;
    width: number;
    height: number;
}

export function useDepthOfField() {
    const adjustments = useEditorStore((s) => s.adjustments);
    const setFinalPreviewUrl = useEditorStore((s) => s.setEditor);
    const setIsGeneratingAi = useProcessStore((s) => s.setIsGeneratingAi);

    const currentPath = useEditorStore((s) => s.selectedImage?.path);
    const isSliderDragging = useEditorStore((s) => s.isSliderDragging);
    const displaySize = useEditorStore((s) => s.displaySize);

    const pendingTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

    const applyBlur = useCallback(
        async (params: DepthOfFieldState, interactive: boolean = false) => {
            if (!currentPath || !params.enabled) return;

            try {
                setIsGeneratingAi(true);
                const result = await invoke<DepthOfFieldResult>(Invokes.ApplyDepthBlur, {
                    jsAdjustments: adjustments,
                    focusDistance: params.focusDistance,
                    blurAmount: params.blurAmount,
                    bokehThreshold: params.bokehThreshold,
                    numRings: params.numRings,
                    samplesPerRing: params.samplesPerRing,
                    bokehShape: params.bokehShape === 'hexagonal' ? 1 : 0,
                    isInteractive: interactive,
                    targetResolution: interactive ? null : { width: displaySize.width, height: displaySize.height },
                    roi: null,
                    computeWaveform: false,
                });

                setFinalPreviewUrl({ finalPreviewUrl: result.preview_url });
            } catch (err) {
                console.error('[DOF] Blur failed:', err);
            } finally {
                setIsGeneratingAi(false);
            }
        },
        [currentPath, adjustments, displaySize],
    );

    // Called on every slider change during drag — debounced for live preview.
    const onSliderChange = useCallback((updates: Partial<DepthOfFieldState>) => {
        _state = { ..._state, ...updates };
        if (pendingTimerRef.current) clearTimeout(pendingTimerRef.current);
        pendingTimerRef.current = setTimeout(() => {
            applyBlur(_state, true); // isInteractive=true → 640×480 preview
        }, 120);
    }, [applyBlur]);

    // Called when slider release — full-res render.
    const onDragEnd = useCallback(() => {
        if (pendingTimerRef.current) clearTimeout(pendingTimerRef.current);
        pendingTimerRef.current = setTimeout(() => {
            applyBlur(_state, false); // Full resolution render
        }, 80);
    }, [applyBlur]);

    const toggleEnabled = useCallback(
        (enabled: boolean) => {
            _state.enabled = enabled;
            if (enabled) {
                applyBlur(_state, isSliderDragging);
            } else {
                // Revert to standard adjustments preview when DOF disabled.
                setFinalPreviewUrl({ finalPreviewUrl: null });
            }
        },
        [applyBlur, isSliderDragging],
    );

    const reset = useCallback(() => {
        _state = { ...DEFAULT_DOF };
    }, []);

    return { state: _state, onSliderChange, onDragEnd, toggleEnabled, reset };
}
