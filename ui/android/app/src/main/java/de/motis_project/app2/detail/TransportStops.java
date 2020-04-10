package de.motis_project.app2.detail;

import android.graphics.drawable.Drawable;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;

import butterknife.BindDrawable;
import butterknife.BindString;
import butterknife.BindView;
import butterknife.ButterKnife;
import butterknife.OnClick;
import de.motis_project.app2.JourneyUtil;
import de.motis_project.app2.R;
import de.motis_project.app2.TimeUtil;
import motis.Connection;

public class TransportStops implements DetailViewHolder {
    private final JourneyUtil.Section section;
    private final View layout;

    private DetailClickHandler activity;
    private final boolean expanded;

    @BindString(R.string.detail_transport_stops_summary) String summaryTemplate;
    @BindString(R.string.detail_transport_stops_summary_no_stopover) String summaryNoStopoverTemplate;
    @BindString(R.string.detail_transport_stops_summary_duration) String summaryNoDurationTemplate;
    @BindString(R.string.stop) String stop;
    @BindString(R.string.stops) String stops;

    @BindView(R.id.detail_transport_stops_upper) ImageView upper;
    @BindView(R.id.detail_transport_stops_lower) ImageView lower;
    @BindView(R.id.detail_transport_stops_summary) TextView summary;
    @BindView(R.id.detail_transport_stops_vertline) View line;

    @BindDrawable(R.drawable.ic_expand_less_black_24dp) Drawable less;
    @BindDrawable(R.drawable.ic_expand_more_black_24dp) Drawable more;

    @OnClick(R.id.detail_transport_stops)
    void onClick() {
        toggleExpand();
    }

    TransportStops(Connection con,
                   JourneyUtil.Section section,
                   ViewGroup parent,
                   LayoutInflater inflater, boolean expanded) {
        this.section = section;
        this.expanded = expanded;
        activity = (DetailClickHandler) inflater.getContext();
        layout = inflater.inflate(R.layout.detail_transport_stops, parent, false);
        ButterKnife.bind(this, layout);

        long clasz = JourneyUtil.getTransport(con, section).clasz();
        JourneyUtil.setBackgroundColor(inflater.getContext(), line, clasz);

        long dep = con.stops(section.from).departure().scheduleTime();
        long arr = con.stops(section.to).arrival().scheduleTime();
        long durationMinutes = (arr - dep) / 60;
        String durationString = durationMinutes == 0 ? "" : String.format(summaryNoDurationTemplate,
                                                                          TimeUtil.formatDuration(
                                                                                  durationMinutes));
        int numStops = section.to - section.from - 1;
        if (numStops == 0) {
            summary.setText(
                    String.format(summaryNoStopoverTemplate, durationString));
        } else {
            summary.setText(
                    String.format(summaryTemplate,
                                  numStops,
                                  numStops == 1 ? stop : stops,
                                  durationString));
        }

        setupIcon(numStops != 0);
    }

    void toggleExpand() {
        if (expanded) {
            activity.contractSection(section);
        } else {
            activity.expandSection(section);
        }
    }

    void setupIcon(boolean visible) {
        int visibility = visible ? View.VISIBLE : View.INVISIBLE;
        upper.setVisibility(visibility);
        lower.setVisibility(visibility);
        if (expanded) {
            upper.setImageDrawable(more);
            lower.setImageDrawable(less);
        } else {
            upper.setImageDrawable(less);
            lower.setImageDrawable(more);
        }
    }

    @Override
    public View getView() {
        return layout;
    }
}
