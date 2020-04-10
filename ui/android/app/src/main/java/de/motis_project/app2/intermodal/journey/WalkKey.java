package de.motis_project.app2.intermodal.journey;

import android.support.annotation.NonNull;
import android.support.annotation.Nullable;

import de.motis_project.app2.ppr.NamedLocation;
import de.motis_project.app2.ppr.profiles.PprSearchOptions;

public class WalkKey {
    private final NamedLocation from;
    private final NamedLocation to;
    private final PprSearchOptions pprSearchOptions;
    private final long duration;
    private final long accessibility;
    private final String mumoType;

    public WalkKey(@NonNull NamedLocation from, @NonNull NamedLocation to,
                   @Nullable PprSearchOptions pprSearchOptions,
                   long duration, long accessibility, @NonNull String mumoType) {
        this.from = from;
        this.to = to;
        this.pprSearchOptions = pprSearchOptions;
        this.duration = duration;
        this.accessibility = accessibility;
        this.mumoType = mumoType;
    }

    public NamedLocation getFrom() {
        return from;
    }

    public NamedLocation getTo() {
        return to;
    }

    public PprSearchOptions getPprSearchOptions() {
        return pprSearchOptions;
    }

    public long getDuration() {
        return duration;
    }

    public long getAccessibility() {
        return accessibility;
    }

    public String getMumoType() {
        return mumoType;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        WalkKey walkKey = (WalkKey) o;

        if (duration != walkKey.duration) return false;
        if (accessibility != walkKey.accessibility) return false;
        if (!from.equalsLocation(walkKey.from)) return false;
        if (!to.equalsLocation(walkKey.to)) return false;
        if (pprSearchOptions != null ? !pprSearchOptions.equals(walkKey.pprSearchOptions) : walkKey.pprSearchOptions != null)
            return false;
        return mumoType.equals(walkKey.mumoType);
    }

    @Override
    public int hashCode() {
        int result = from.hashCode();
        result = 31 * result + to.hashCode();
        result = 31 * result + (pprSearchOptions != null ? pprSearchOptions.hashCode() : 0);
        result = 31 * result + (int) (duration ^ (duration >>> 32));
        result = 31 * result + (int) (accessibility ^ (accessibility >>> 32));
        result = 31 * result + mumoType.hashCode();
        return result;
    }

    @Override
    public String toString() {
        return "WalkKey{" +
                "from=" + from +
                ", to=" + to +
                ", pprSearchOptions=" + pprSearchOptions +
                ", duration=" + duration +
                ", accessibility=" + accessibility +
                ", mumoType='" + mumoType + '\'' +
                '}';
    }
}
