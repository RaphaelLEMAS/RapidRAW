import { useCallback, useRef } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { Invokes } from '../components/ui/AppProperties';
import { useEditorStore } from '../store/useEditorStore';

export interface DepthOfFieldState {
    enabled: boolean;
    focusDistance: number;  // 0.0 (near) to 1.0 (far) — which depth plane stays sharp
    blurAmount: number;     // Max blur radius in pixels, 0-40
    bokehThreshold: number; // Min depth offset for blur, 0.0-0.2 (focal range width)
    numRings: number;       // Concentric rings: 3-5 (quality vs speed tradeoff)
    samplesPerRing: number; // Samples per ring: 6-10
    bokehShape: 'circular' | 'hexagonal'; // Bokeh shape mode: 0=circular, 1=hexagonal
}

// Default values — kept in sync with useEditorStore initialization
const DEFAULT_DOF = {
    enabled: false,
    focusDistance: 0.5,
    blurAmount: 8,
    bokehThreshold: 0.03,
    numRings: 3,
    samplesPerRing: 7,
    bokehShape: 'circular',
} as const;

interface DepthOfFieldResult {
    preview_url: string;
    width: number;
    height: number;
}

export function useDepthOfField() {
    const dof = useEditorStore((s) => ({
        enabled: s.dofEnabled,
        focusDistance: s.dofFocusDistance,
        blurAmount: s.dofBlurAmount,
        bokehThreshold: s.dofBokehThreshold,
        numRings: s.dofNumRings,
        samplesPerRing: s.dofSamplesPerRing,
        bokehShape: s.dofBokehShape,
    }));

    const setEditor = useEditorStore((s) => s.setEditor);
    const adjustments = useEditorStore((s) => s.adjustments);
    const currentPath = useEditorStore((s) => s.selectedImage?.path);
    const isSliderDragging = useEditorStore((s) => s.isSliderDragging);
    const displaySize = useEditorStore((s) => s.displaySize);

    // Fix: isGeneratingAi lives in useEditorStore, NOT useProcessStore
    const updateIsGeneratingAi = useCallback((value: boolean) => {
        setEditor({ isGeneratingAi: value });
    }, [setEditor]);

    const pendingTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

    const applyBlur = useCallback(
        async (params: DepthOfFieldState, interactive: boolean = false) => {
            if (!currentPath || !params.enabled) return;

            try {
                updateIsGeneratingAi(true);
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

                setEditor({ finalPreviewUrl: result.preview_url });
            } catch (err) {
                console.error('[DOF] Blur failed:', err);
            } finally {
                updateIsGeneratingAi(false);
            }
        },
        [currentPath, adjustments, displaySize, setEditor, updateIsGeneratingAi],
    );

    const latestStateRef = useRef(dof);
    latestStateRef.current = dof;

    // Called on every slider change during drag — debounced for live preview.
    const onSliderChange = useCallback((updates: Partial<DepthOfFieldState>) => {
        setEditor((prev) => ({
            dofEnabled: 'enabled' in updates ? updates.enabled : prev.dofEnabled,
            dofFocusDistance: 'focusDistance' in updates ? updates.focusDistance : prev.dofFocusDistance,
            dofBlurAmount: 'blurAmount' in updates ? updates.blurAmount : prev.dofBlurAmount,
            dofBokehThreshold: 'bokehThreshold' in updates ? updates.bokehThreshold : prev.dofBokehThreshold,
            dofNumRings: 'numRings' in updates ? updates.numRings : prev.dofNumRings,
            dofSamplesPerRing: 'samplesPerRing' in updates ? updates.samplesPerRing : prev.dofSamplesPerRing,
            dofBokehShape: 'bokehShape' in updates ? updates.bokehShape : prev.dofBokehShape,
        }));

        if (pendingTimerRef.current) clearTimeout(pendingTimerRef.current);
        pendingTimerRef.current = setTimeout(() => {
            // Read from ref to avoid stale closure values
            const current = latestStateRef.current;
            applyBlur(current, true); // isInteractive=true → 640×480 preview
        }, 120);
    }, [applyBlur]);

    // Called when slider release — full-res render.
    const onDragEnd = useCallback(() => {
        if (pendingTimerRef.current) clearTimeout(pendingTimerRef.current);
        pendingTimerRef.current = setTimeout(() => {
            applyBlur(latestStateRef.current, false); // Full resolution render
        }, 80);
    }, [applyBlur]);

    const toggleEnabled = useCallback(
        (enabled: boolean) => {
            setEditor({ dofEnabled: enabled });
            if (enabled) {
                applyBlur(dof, isSliderDragging);
            } else {
                // Revert to standard adjustments preview when DOF disabled.
                setEditor({ finalPreviewUrl: null });
            }
        },
        [applyBlur, isSliderDragging, dof],
    );

    const reset = useCallback(() => {
        setEditor((prev) => ({
            dofEnabled: DEFAULT_DOF.enabled,
            dofFocusDistance: DEFAULT_DOF.focusDistance,
            dofBlurAmount: DEFAULT_DOF.blurAmount,
            dofBokehThreshold: DEFAULT_DOF.bokehThreshold,
            dofNumRings: DEFAULT_DOF.numRings,
            dofSamplesPerRing: DEFAULT_DOF.samplesPerRing,
            dofBokehShape: DEFAULT_DOF.bokehShape,
        }));
    }, [setEditor]);

    return { state: dof, onSliderChange, onDragEnd, toggleEnabled, reset };
}
