package de.motis_project.app2.detail;

import de.motis_project.app2.JourneyUtil;
import de.motis_project.app2.ppr.route.StepInfo;
import motis.Stop;

interface DetailClickHandler {
    void expandSection(JourneyUtil.Section section);
    void contractSection(JourneyUtil.Section section);
    void refreshSection(JourneyUtil.Section section);

    void transportStopClicked(Stop stop);
    void walkStepClicked(StepInfo stepInfo);
}
