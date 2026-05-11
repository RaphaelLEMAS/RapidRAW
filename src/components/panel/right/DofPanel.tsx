import Slider from '../../ui/Slider';
import Switch from '../../ui/Switch';
import Text from '../../ui/Text';
import { TEXT_COLOR_KEYS, TextVariants, TextWeights } from '../../../types/typography';
import { useDepthOfField } from '../../../hooks/useDepthOfField';

interface DofPanelProps {
    onDragStateChange?: (isDragging: boolean) => void;
}

export default function DofPanel({ onDragStateChange }: DofPanelProps) {
    const { state, onSliderChange, onDragEnd, toggleEnabled } = useDepthOfField();

    return (
        <div className="flex flex-col gap-5 p-2">
            {/* Toggle */}
            <div className="flex items-center justify-between pb-1 border-b border-[var(--border-color)]">
                <Text variant={TextVariants.label} color={TEXT_COLOR_KEYS['text-primary']}>
                    Portrait Blur
                </Text>
                <Switch checked={state.enabled} onChange={toggleEnabled} />
            </div>

            {/* Focus Distance */}
            <div className="space-y-2">
                <div className="flex justify-between text-xs">
                    <span className="text-[var(--text-secondary)]">Near</span>
                    <Text variant={TextVariants.small} color={TEXT_COLOR_KEYS['text-primary']} weight={TextWeights.medium}>
                        Focus Distance
                    </Text>
                    <span className="text-[var(--text-secondary)]">Far</span>
                </div>
                <Slider
                    min={0} max={1} step={0.01}
                    value={state.focusDistance}
                    onDragStateChange={onDragStateChange}
                    onChange={(e) => onSliderChange({ focusDistance: parseFloat(e.target.value) })}
                    onDragEnd={onDragEnd}
                />
            </div>

            {/* Blur Amount */}
            <div className="space-y-2">
                <div className="flex justify-between text-xs">
                    <span className="text-[var(--text-secondary)]">Subtle</span>
                    <Text variant={TextVariants.small} color={TEXT_COLOR_KEYS['text-primary']} weight={TextWeights.medium}>
                        Blur Strength
                    </Text>
                    <span className="text-[var(--text-secondary)]">Strong</span>
                </div>
                <Slider
                    min={0} max={40} step={1}
                    value={state.blurAmount}
                    onDragStateChange={onDragStateChange}
                    onChange={(e) => onSliderChange({ blurAmount: parseFloat(e.target.value) })}
                    onDragEnd={onDragEnd}
                />
            </div>

            {/* Bokeh Threshold / Focal Range */}
            <div className="space-y-2">
                <div className="flex justify-between text-xs">
                    <span className="text-[var(--text-secondary)]">Narrow</span>
                    <Text variant={TextVariants.small} color={TEXT_COLOR_KEYS['text-primary']} weight={TextWeights.medium}>
                        Focal Range
                    </Text>
                    <span className="text-[var(--text-secondary)]">Wide</span>
                </div>
                <Slider
                    min={0.0} max={0.2} step={0.005}
                    value={state.bokehThreshold}
                    onDragStateChange={onDragStateChange}
                    onChange={(e) => onSliderChange({ bokehThreshold: parseFloat(e.target.value) })}
                    onDragEnd={onDragEnd}
                />
            </div>

            {/* Bokeh Shape selector */}
            <div className="space-y-2">
                <Text variant={TextVariants.small} color={TEXT_COLOR_KEYS['text-primary']} weight={TextWeights.medium}>
                    Bokeh Shape
                </Text>
                <div className="flex gap-2">
                    {(['circular', 'hexagonal'] as const).map((shape) => (
                        <button
                            key={shape}
                            type="button"
                            onClick={() => onSliderChange({ bokehShape: shape })}
                            className={`px-3 py-1.5 text-xs rounded-md transition-colors ${
                                state.bokehShape === shape
                                    ? 'bg-[var(--accent)] text-white'
                                    : 'bg-[var(--surface-secondary)] text-[var(--text-secondary)] hover:bg-[var(--surface-hover)]'
                            }`}
                        >
                            {shape.charAt(0).toUpperCase() + shape.slice(1)}
                        </button>
                    ))}
                </div>
            </div>

            {/* Hint */}
            <Text variant={TextVariants.caption} color={TEXT_COLOR_KEYS['text-secondary']} className="pt-2 border-t border-[var(--border-color)]">
                Generate a depth map first via the AI panel, then adjust these controls.
            </Text>
        </div>
    );
}
