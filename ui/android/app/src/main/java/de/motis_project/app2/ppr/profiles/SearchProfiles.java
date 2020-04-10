package de.motis_project.app2.ppr.profiles;

import java.util.ArrayList;

public class SearchProfiles {
    public static final int DEFAULT_MAX_DURATION = 20;

    public final ArrayList<SearchProfile> profiles;

    public SearchProfiles() {
        profiles = new ArrayList<>();
        profiles.add(new SearchProfile("default", "Standard"));
        profiles.add(new SearchProfile("accessibility1", "Auch leichte Wege"));
        profiles.add(new SearchProfile("wheelchair", "Rollstuhl"));
        profiles.add(new SearchProfile("elevation", "Weniger Steigung"));
    }

    public SearchProfile getById(String id, SearchProfile def) {
        if (id != null) {
            for (SearchProfile p : profiles) {
                if (p.getId().equals(id)) {
                    return p;
                }
            }
        }
        return def;
    }

    public SearchProfile getDefault() {
        return profiles.get(0);
    }

    public PprSearchOptions getDefaultSearchOptions() {
        return new PprSearchOptions(getDefault(), DEFAULT_MAX_DURATION);
    }
}
