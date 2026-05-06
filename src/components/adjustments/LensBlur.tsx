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

// Standard photographic full f-stops (√2 geometric progression from f/1.0 to f/32)
const STANDARD_FSTOPS = [1.0, 1.4, 2.0, 2.8, 4.0, 5.6, 8.0, 11, 16, 22, 32];

// DISCRETE_STEPS mode: slider value is an INDEX into STANDARD_FSTOPS
const FSTOP_SLIDER_MAX = (STANDARD_FSTOPS.length - 1) * 10; // 100 (= index 10, f/32)
const FSTOP_SLIDER_MIN = 0; // index 0, f/1.0

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

// Map slider numeric value (0..100) to discrete f-stop via √2 progression
function sliderToFStop(sliderValue: number): number {
  const clamped = clamp(sliderValue, FSTOP_SLIDER_MIN, FSTOP_SLIDER_MAX);
  // Each full stop = 10 slider units; index = floor(clamped / 10)
  const index = Math.round(clamped / 10);
  return STANDARD_FSTOPS[index];
}

// Display label from stored f-stop value (reads the standard array directly)
function formatFStop(fstop: number): string {
  // Find which standard stop this value corresponds to and display it cleanly
  const index = STANDARD_FSTOPS.indexOf(Math.round(fstop * 10) / 10);
  if (index !== -1) {
    return `f/${STANDARD_FSTOPS[index]}`;
  }
  // Fallback: round to nearest standard stop
  let closest = STANDARD_FSTOPS[0];
  for (const s of STANDARD_FSTOPS) {
    if (Math.abs(s - fstop) < Math.abs(closest - fstop)) {
      closest = s;
    }
  }
  return `f/${closest}`;
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

  // Aperture stored as f-stop index * 10 for sub-step granularity in slider
  const lensFStopIndex = adjustments.lensFStop || 6; // default index=6 → f/5.6
  const lensFStopValue = sliderToFStop(lensFStopIndex);

  const bokehShape = adjustments.bokehShape || 0;
  const lensAnisotropy = adjustments.lensAnisotropy || 0;
  const lensFalloffEnabled = (adjustments.lensFalloffEnabled ?? 100) > 50;
  const lensFalloffAmount = adjustments.lensFalloffAmount || 30;
  const falloffMode = adjustments.falloffMode || 0;

  // Lens blur amount and radius (retrieved after bokehShape for readability)
  const lensBlurAmount = adjustments.lensBlurAmount || 0;
  const lensRadius = adjustments.lensRadius || 50.0;

  const isCircular = bokehShape === 0;

  return (
    <div className="space-y-2">
      <Slider
        label="Blur Amount"
        max={100}
        min={0}
        onChange={(e: any) => handleLensBlurChange(CreativeAdjustment.LensBlurAmount, parseInt(e.target.value, 10))}
        step={1}
        value={lensBlurAmount}
        suffix="%"
      />

      <Slider
        label={`Aperture (F): ${formatFStop(lensFStopValue)}`}
        max={FSTOP_SLIDER_MAX}
        min={FSTOP_SLIDER_MIN}
        step={10}
        onChange={(e: any) => {
          const raw = clamp(parseFloat(e.target.value), FSTOP_SLIDER_MIN, FSTOP_SLIDER_MAX);
          handleLensBlurChange(CreativeAdjustment.LensFStop, raw);
        }}
        value={lensFStopIndex}
      />

      <Slider
        label="Lens Radius"
        max={50}
        min={0}
        onChange={(e: any) => {
          const clamped = clamp(parseInt(e.target.value, 10), 0, 50);
          handleLensBlurChange(CreativeAdjustment.LensRadius, clamped);
        }}
        step={1}
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
        label={`Anisotropy: ${lensAnisotropy >= 0 ? '+' : ''}${Math.round(lensAnisotropy)}°`}
        max={90}
        min={-90}
        onChange={(e: any) => handleLensBlurChange(CreativeAdjustment.LensAnisotropy, parseInt(e.target.value, 10))}
        step={1}
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
              onChange={(e: any) => handleLensBlurChange(CreativeAdjustment.LensFalloffAmount, parseInt(e.target.value, 10))}
              step={1}
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
