import Slider from '../ui/Slider';
import Dropdown from '../ui/Dropdown';
import Switch from '../ui/Switch';
import Text from '../ui/Text';
import { TextVariants } from '../../types/typography';
import { Adjustments, CreativeAdjustment } from '../../utils/adjustments';

interface LensBlurProps {
  adjustments: Partial<Adjustments>;
  setAdjustments(adjustments: Partial<Adjustments>): any;
}

const BOKEH_SHAPES = [
  { label: 'Circular', value: 0 },
  { label: 'Hexagonal', value: 1 },
  { label: 'Octagonal', value: 2 },
  { label: 'Custom SVG', value: 3 },
];

const FALLOFF_MODES = [
  { label: 'Linear', value: 0 },
  { label: 'Exponential', value: 1 },
];

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

// Format slider display values for continuous control
function fmtBlur(v: number): string {
  if (v >= 10) return String(Math.round(v));
  const rounded = parseFloat(v.toFixed(1));
  return String(rounded);
}

export default function LensBlur({ adjustments, setAdjustments }: LensBlurProps) {
  const handleLensBlurChange = (key: string, value: number | boolean) => {
    if (typeof value === 'boolean') {
      const numVal = value ? 100 : 0;
      setAdjustments((prev: Partial<Adjustments>) => ({ ...prev, [key]: numVal }));
    } else {
      setAdjustments((prev: Partial<Adjustments>) => ({ ...prev, [key]: value }));
    }
  };

  const handleDropdownChange = (key: string) => (value: number | null) => {
    if (value !== null) {
      setAdjustments((prev: Partial<Adjustments>) => ({ ...prev, [key]: value }));
    }
  };

  // Continuous float values — no snapping or interpolation
  const lensBlurAmount = adjustments.lensBlurAmount || 0;
  const lensFStop = clamp(adjustments.lensFStop ?? 50, 0, 100);     // abstract intensity [0..100]
  const lensRadius = clamp(adjustments.lensRadius || 0, 0, 50);
  const bokehShape = adjustments.bokehShape || 0;
  const lensAnisotropy = clamp(adjustments.lensAnisotropy ?? 0, -90, 90);
  const lensFalloffEnabled = (adjustments.lensFalloffEnabled ?? 100) > 50;
  const lensFalloffAmount = clamp(adjustments.lensFalloffAmount || 30, 0, 100);
  const falloffMode = adjustments.falloffMode || 0;

  const isCircular = bokehShape === 0;

  return (
    <div className="space-y-2">
      <Slider
        label="Blur Amount"
        max={100}
        min={0}
        onChange={(e: any) => handleLensBlurChange(CreativeAdjustment.LensBlurAmount, parseFloat(e.target.value))}
        step={0}
        value={lensBlurAmount}
        suffix="%"
      />

      <Slider
        label={`Bokeh Intensity: ${fmtBlur(lensFStop)}`}
        max={100}
        min={0}
        onChange={(e: any) => {
          const raw = clamp(parseFloat(e.target.value), 0, 100);
          handleLensBlurChange(CreativeAdjustment.LensFStop, raw);
        }}
        step={0}
        value={lensFStop}
      />

      <Slider
        label="Lens Radius"
        max={50}
        min={0}
        onChange={(e: any) => {
          const clamped = clamp(parseFloat(e.target.value), 0, 50);
          handleLensBlurChange(CreativeAdjustment.LensRadius, clamped);
        }}
        step={0}
        value={lensRadius}
        suffix="px"
      />

      <Dropdown<number>
        className=""
        options={BOKEH_SHAPES}
        placeholder="Select bokeh shape"
        value={bokehShape as number}
        onChange={handleDropdownChange(CreativeAdjustment.BokehShape)}
      />

      <Slider
        label={`Anisotropy: ${lensAnisotropy >= 0 ? '+' : ''}${fmtBlur(lensAnisotropy)}°`}
        max={90}
        min={-90}
        onChange={(e: any) => {
          const clamped = clamp(parseFloat(e.target.value), -90, 90);
          handleLensBlurChange(CreativeAdjustment.LensAnisotropy, clamped);
        }}
        step={0}
        value={lensAnisotropy}
      />

      <div className="space-y-2">
        <Switch
          checked={lensFalloffEnabled}
          label="Optical Falloff"
          onChange={(val) => handleLensBlurChange(CreativeAdjustment.LensFalloffEnabled, val)}
        />

        {lensFalloffEnabled && (
          <>
            <Slider
              label="Falloff Amount"
              max={100}
              min={0}
              onChange={(e: any) => handleLensBlurChange(CreativeAdjustment.LensFalloffAmount, parseFloat(e.target.value))}
              step={0}
              value={lensFalloffAmount}
              suffix="%"
            />

            <Dropdown<number>
              className=""
              options={FALLOFF_MODES}
              placeholder="Select falloff mode"
              value={falloffMode as number}
              onChange={handleDropdownChange(CreativeAdjustment.FalloffMode)}
            />
          </>
        )}
      </div>

      {isCircular && (
        <Text variant={TextVariants.caption} className="text-text-tertiary italic">
          Circular bokeh produces smooth, round out-of-focus highlights. Anisotropy has no visual effect on perfectly circular shapes but is preserved for preset compatibility.
        </Text>
      )}

      {!isCircular && (
        <Text variant={TextVariants.caption} className="text-text-tertiary italic">
          Polygonal bokeh simulates diaphragm blade geometry. Rotation at 30-degree increments aligns shapes symmetrically for starburst highlight effects.
        </Text>
      )}
    </div>
  );
}
