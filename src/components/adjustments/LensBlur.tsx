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

function formatFStop(fstop: number): string {
  const stops = [1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.5, 2.8, 3.2, 3.5, 4.0, 4.5, 5.0, 5.6, 6.3, 7.1, 8.0, 9.0, 10.0, 11.0, 13.0, 14.0, 16.0, 18.0, 20.0, 22.0];
  let closest = stops[5]; // f/2.8 default
  for (const s of stops) {
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

  const lensBlurAmount = adjustments.lensBlurAmount || 0;
  const lensFStop = adjustments.lensFStop || 28.0;
  const lensRadius = adjustments.lensRadius || 50.0;
  const bokehShape = adjustments.bokehShape || 0;
  const lensAnisotropy = adjustments.lensAnisotropy || 0;
  const lensFalloffEnabled = (adjustments.lensFalloffEnabled ?? 100) > 50;
  const lensFalloffAmount = adjustments.lensFalloffAmount || 30;
  const falloffMode = adjustments.falloffMode || 0;

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
        label={`Aperture (F): ${formatFStop(lensFStop)}`}
        max={100}
        min={0}
        onChange={(e: any) => handleLensBlurChange(CreativeAdjustment.LensFStop, parseFloat(e.target.value))}
        step={1}
        value={lensFStop}
      />

      <Slider
        label="Lens Radius"
        max={50}
        min={0}
        onChange={(e: any) => handleLensBlurChange(CreativeAdjustment.LensRadius, parseInt(e.target.value, 10))}
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
