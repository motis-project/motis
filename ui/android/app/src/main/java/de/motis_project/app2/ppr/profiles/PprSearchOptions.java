package de.motis_project.app2.ppr.profiles;

import java.util.Objects;

public class PprSearchOptions {
    public SearchProfile profile;
    public int maxDuration; // minutes

    public PprSearchOptions(SearchProfile profile, int maxDuration) {
        this.profile = profile;
        this.maxDuration = maxDuration;
    }

    @Override
    public String toString() {
        return "PprSearchOptions{" +
                "profile=" + profile +
                ", maxDuration=" + maxDuration +
                '}';
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        PprSearchOptions that = (PprSearchOptions) o;
        return maxDuration == that.maxDuration &&
                profile.equals(that.profile);
    }

    @Override
    public int hashCode() {
        return Objects.hash(profile, maxDuration);
    }
}
