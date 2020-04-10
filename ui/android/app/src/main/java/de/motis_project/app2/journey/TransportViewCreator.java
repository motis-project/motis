package de.motis_project.app2.journey;

import android.content.Context;
import android.view.LayoutInflater;
import android.widget.LinearLayout;
import android.widget.TextView;

import java.util.List;

import de.motis_project.app2.JourneyUtil;
import de.motis_project.app2.R;

public class TransportViewCreator {
    public static void addTransportViews(
            List<JourneyUtil.DisplayTransport> transports,
            LayoutInflater inflater,
            LinearLayout target) {
        target.removeAllViews();

        JourneySummaryViewHolder.ViewMode viewMode = getViewMode(transports);
        for (int i = 0; i < transports.size(); i++) {
            JourneyUtil.DisplayTransport t = transports.get(i);
            TextView view = (TextView) inflater.inflate(R.layout.journey_item_transport_train,
                                                        target, false);

            Context context = inflater.getContext();
            JourneyUtil.tintBackground(context, view, t.clasz);
            JourneyUtil.setIcon(context, view, t.clasz);

            if (viewMode != JourneySummaryViewHolder.ViewMode.OFF) {
                view.setText(viewMode == JourneySummaryViewHolder.ViewMode.SHORT ? t.shortName : t.longName);
            }
            target.addView(view);

            if (i != transports.size() - 1) {
                target.addView(inflater.inflate(
                        R.layout.journey_item_transport_separator,
                        target,
                        false));
            }
        }
    }

    static private JourneySummaryViewHolder.ViewMode getViewMode(
            List<JourneyUtil.DisplayTransport> transports) {
        final long MAX_SIZE = 25;
        final long transportsTerm = 3 * transports.size();
        if (getTextLengthSum(transports, JourneySummaryViewHolder.ViewMode.LONG) + transportsTerm <=
                MAX_SIZE) {
            return JourneySummaryViewHolder.ViewMode.LONG;
        } else if (getTextLengthSum(transports, JourneySummaryViewHolder.ViewMode.SHORT) + transportsTerm <=
                MAX_SIZE) {
            return JourneySummaryViewHolder.ViewMode.SHORT;
        } else {
            return JourneySummaryViewHolder.ViewMode.OFF;
        }
    }

    static private int getTextLengthSum(List<JourneyUtil.DisplayTransport> transports,
                                        JourneySummaryViewHolder.ViewMode mode) {
        StringBuffer buf = new StringBuffer();
        for (JourneyUtil.DisplayTransport t : transports) {
            buf.append(mode == JourneySummaryViewHolder.ViewMode.LONG ? t.longName : t.shortName);
        }
        return buf.toString().length();
    }
}
