// Generated claw INNER opening lookup. CAD data is theoretical, not measured.
// Not yaw/TCP parameters. Install together with the matching PC JSON.
#ifndef CLAW_PREOPEN_PROFILE_H
#define CLAW_PREOPEN_PROFILE_H
#include <cmath>
#include <cstddef>
namespace ClawPreopen {
struct Point { float gap_mm; float servo_deg; };
static constexpr bool kConfigured = true;
static constexpr bool kCadModel = true;
static constexpr float kClosedAngleDeg = -35.000000000f;
static constexpr float kStandbyAngleDeg = -95.000000000f;

// The lookup table below was generated with the old CAD reference where
// -15 deg was treated as the closed reference. The real machine now uses
// -35 deg as the physical closed command, so new effective opening values
// are rebased before looking up the old CAD table.
static constexpr float kCadReferenceClosedAngleDeg = -15.000000000f;
static constexpr float kClosedBaselineGapMm = 21.020967118f;
static constexpr Point kPoints[] = {
    {0.866004384f, -16.000000000f},
    {1.399094177f, -16.500000000f},
    {1.932244255f, -17.000000000f},
    {2.465452280f, -17.500000000f},
    {2.998713217f, -18.000000000f},
    {3.532019558f, -18.500000000f},
    {4.065361523f, -19.000000000f},
    {4.598727250f, -19.500000000f},
    {5.132102958f, -20.000000000f},
    {5.665473106f, -20.500000000f},
    {6.198820530f, -21.000000000f},
    {6.732126569f, -21.500000000f},
    {7.265371187f, -22.000000000f},
    {7.798533072f, -22.500000000f},
    {8.331589741f, -23.000000000f},
    {8.864517625f, -23.500000000f},
    {9.397292150f, -24.000000000f},
    {9.929887817f, -24.500000000f},
    {10.462278268f, -25.000000000f},
    {10.994436348f, -25.500000000f},
    {11.526334169f, -26.000000000f},
    {12.057943161f, -26.500000000f},
    {12.589234120f, -27.000000000f},
    {13.120177260f, -27.500000000f},
    {13.650742251f, -28.000000000f},
    {14.180898261f, -28.500000000f},
    {14.710613993f, -29.000000000f},
    {15.239857718f, -29.500000000f},
    {15.768597310f, -30.000000000f},
    {16.296800272f, -30.500000000f},
    {16.824433768f, -31.000000000f},
    {17.351464645f, -31.500000000f},
    {17.877859461f, -32.000000000f},
    {18.403584505f, -32.500000000f},
    {18.928605820f, -33.000000000f},
    {19.452889222f, -33.500000000f},
    {19.976400322f, -34.000000000f},
    {20.499104538f, -34.500000000f},
    {21.020967118f, -35.000000000f},
    {21.541953153f, -35.500000000f},
    {22.062027591f, -36.000000000f},
    {22.581155254f, -36.500000000f},
    {23.099300848f, -37.000000000f},
    {23.616428980f, -37.500000000f},
    {24.132504167f, -38.000000000f},
    {24.647490848f, -38.500000000f},
    {25.161353395f, -39.000000000f},
    {25.674056125f, -39.500000000f},
    {26.185563310f, -40.000000000f},
    {26.695839183f, -40.500000000f},
    {27.204847951f, -41.000000000f},
    {27.712553804f, -41.500000000f},
    {28.218920921f, -42.000000000f},
    {28.723913477f, -42.500000000f},
    {29.227495657f, -43.000000000f},
    {29.729631657f, -43.500000000f},
    {30.230285693f, -44.000000000f},
    {30.729422011f, -44.500000000f},
    {31.227004889f, -45.000000000f},
    {31.722998649f, -45.500000000f},
    {32.217367656f, -46.000000000f},
    {32.710076333f, -46.500000000f},
    {33.201089159f, -47.000000000f},
    {33.690370680f, -47.500000000f},
    {34.177885510f, -48.000000000f},
    {34.663598342f, -48.500000000f},
    {35.147473949f, -49.000000000f},
    {35.629477191f, -49.500000000f},
    {36.109573018f, -50.000000000f},
    {36.587726477f, -50.500000000f},
    {37.063902718f, -51.000000000f},
    {37.538066994f, -51.500000000f},
    {38.010184671f, -52.000000000f},
    {38.480221226f, -52.500000000f},
    {38.948142260f, -53.000000000f},
    {39.413913494f, -53.500000000f},
    {39.877500777f, -54.000000000f},
    {40.338870090f, -54.500000000f},
    {40.797987551f, -55.000000000f},
    {41.254819415f, -55.500000000f},
    {41.709332082f, -56.000000000f},
    {42.161492101f, -56.500000000f},
    {42.611266168f, -57.000000000f},
    {43.058621137f, -57.500000000f},
    {43.503524018f, -58.000000000f},
    {43.945941986f, -58.500000000f},
    {44.385842377f, -59.000000000f},
    {44.823192699f, -59.500000000f},
    {45.257960631f, -60.000000000f},
    {45.690114029f, -60.500000000f},
    {46.119620924f, -61.000000000f},
    {46.546449533f, -61.500000000f},
    {46.970568256f, -62.000000000f},
    {47.391945681f, -62.500000000f},
    {47.810550590f, -63.000000000f},
    {48.226351958f, -63.500000000f},
    {48.639318955f, -64.000000000f},
    {49.049420956f, -64.500000000f},
    {49.456627536f, -65.000000000f},
    {49.860908479f, -65.500000000f},
    {50.262233775f, -66.000000000f},
    {50.660573629f, -66.500000000f},
    {51.055898460f, -67.000000000f},
    {51.448178904f, -67.500000000f},
    {51.837385817f, -68.000000000f},
    {52.223490279f, -68.500000000f},
    {52.606463596f, -69.000000000f},
    {52.986277300f, -69.500000000f},
    {53.362903158f, -70.000000000f},
    {53.736313167f, -70.500000000f},
    {54.106479560f, -71.000000000f},
    {54.473374811f, -71.500000000f},
    {54.836971634f, -72.000000000f},
    {55.197242985f, -72.500000000f},
    {55.554162068f, -73.000000000f},
    {55.907702333f, -73.500000000f},
    {56.257837484f, -74.000000000f},
    {56.604541474f, -74.500000000f},
    {56.947788514f, -75.000000000f},
    {57.287553073f, -75.500000000f},
    {57.623809876f, -76.000000000f},
    {57.956533916f, -76.500000000f},
    {58.285700444f, -77.000000000f},
    {58.611284983f, -77.500000000f},
    {58.933263321f, -78.000000000f},
    {59.251611517f, -78.500000000f},
    {59.566305905f, -79.000000000f},
    {59.877323092f, -79.500000000f},
    {60.184639962f, -80.000000000f},
    {60.488233680f, -80.500000000f},
    {60.788081690f, -81.000000000f},
    {61.084161718f, -81.500000000f},
    {61.376451778f, -82.000000000f},
    {61.664930170f, -82.500000000f},
    {61.949575480f, -83.000000000f},
    {62.230366587f, -83.500000000f},
    {62.507282664f, -84.000000000f},
    {62.780303174f, -84.500000000f},
    {63.049407881f, -85.000000000f},
    {63.314576843f, -85.500000000f},
    {63.575790420f, -86.000000000f},
    {63.833029273f, -86.500000000f},
    {64.086274366f, -87.000000000f},
    {64.335506967f, -87.500000000f},
    {64.580708651f, -88.000000000f},
    {64.821861303f, -88.500000000f},
    {65.058947116f, -89.000000000f},
    {65.291948594f, -89.500000000f},
    {65.520848555f, -90.000000000f},
    {65.745630132f, -90.500000000f},
    {65.966276773f, -91.000000000f},
    {66.182772244f, -91.500000000f},
    {66.395100631f, -92.000000000f},
    {66.603246340f, -92.500000000f},
    {66.807194099f, -93.000000000f},
    {67.006928959f, -93.500000000f},
    {67.202436297f, -94.000000000f},
    {67.393701817f, -94.500000000f},
    {67.580711548f, -95.000000000f},
};
static constexpr std::size_t kCount = sizeof(kPoints) / sizeof(kPoints[0]);

// Real effective INNER opening range after rebasing -35 deg to the closed point.
// Keep a small non-zero minimum command for safety.
static constexpr float kMinGapMm = 0.900000000f;
static constexpr float kMaxGapMm = 46.500000000f;

inline bool ToAngle(float gap_mm, float &angle_deg)
{
    // ------------------------------------------------------------------------
    // 1. Validate the NEW real-machine effective opening command.
    // ------------------------------------------------------------------------
    if (!kConfigured ||
        kCount < 3 ||
        !std::isfinite(gap_mm) ||
        gap_mm < kMinGapMm ||
        gap_mm > kMaxGapMm)
    {
        return false;
    }

    // ------------------------------------------------------------------------
    // 2. Convert the NEW effective opening back to the OLD CAD table basis.
    //
    // Old CAD table:
    //     -35 deg -> 21.020967118 mm
    //
    // Real machine:
    //     -35 deg -> 0 mm closed reference
    //
    // Therefore:
    //     cad_gap = real_effective_gap + 21.020967118
    // ------------------------------------------------------------------------
    const float cad_gap_mm =
        gap_mm + kClosedBaselineGapMm;

    const float direction =
        kPoints[kCount - 1].servo_deg -
        kPoints[0].servo_deg;

    // ------------------------------------------------------------------------
    // 3. Validate the OLD CAD table itself.
    //
    // IMPORTANT:
    // The table contains points from -16 deg to -95 deg, so table validation
    // must use the OLD CAD reference (-15 deg), not the new real closed angle
    // (-35 deg). Otherwise all points from -16 to -34.5 would be rejected.
    // ------------------------------------------------------------------------
    for (std::size_t i = 0; i < kCount; ++i)
    {
        const Point &p = kPoints[i];

        if (!std::isfinite(p.gap_mm) ||
            p.gap_mm <= 0.0f ||
            p.gap_mm > 120.0f ||
            !std::isfinite(p.servo_deg) ||
            p.servo_deg < -135.0f ||
            p.servo_deg > 135.0f)
        {
            return false;
        }

        if (kCadModel &&
            (p.servo_deg < kStandbyAngleDeg ||
             p.servo_deg >= kCadReferenceClosedAngleDeg ||
             direction >= 0.0f))
        {
            return false;
        }

        if (i > 0)
        {
            if (p.gap_mm <= kPoints[i - 1].gap_mm)
            {
                return false;
            }

            if ((p.servo_deg - kPoints[i - 1].servo_deg) *
                    direction <=
                0.0f)
            {
                return false;
            }
        }
    }

    // ------------------------------------------------------------------------
    // 4. Interpolate in the OLD CAD table using the rebased CAD gap.
    // ------------------------------------------------------------------------
    for (std::size_t i = 1; i < kCount; ++i)
    {
        if (cad_gap_mm <= kPoints[i].gap_mm)
        {
            const Point &a = kPoints[i - 1];
            const Point &b = kPoints[i];

            const float candidate =
                a.servo_deg +
                (b.servo_deg - a.servo_deg) *
                    (cad_gap_mm - a.gap_mm) /
                    (b.gap_mm - a.gap_mm);

            if (!std::isfinite(candidate))
            {
                return false;
            }

            // Pre-open must remain on the OPEN side of the final closed angle.
            // Since more-negative angles are more open:
            //     preopen angle must be < -35 deg.
            if (candidate >= kClosedAngleDeg)
            {
                return false;
            }

            angle_deg = candidate;
            return true;
        }
    }

    return false;
}

} // namespace ClawPreopen
#endif
