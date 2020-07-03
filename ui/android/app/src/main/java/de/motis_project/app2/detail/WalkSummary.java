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
import motis.Walk;

public class WalkSummary implements DetailViewHolder {
    private final JourneyUtil.Section section;
    private final View layout;

    private DetailClickHandler activity;
    private final boolean expanded;

    @BindString(R.string.detail_walk_summary)
    String summaryTemplate;
    @BindString(R.string.detail_walk_summary_duration)
    String summaryDurationTemplate;

    @BindView(R.id.detail_walk_summary_upper)
    ImageView upper;
    @BindView(R.id.detail_walk_summary_lower)
    ImageView lower;
    @BindView(R.id.detail_walk_summary_text)
    TextView summary;
    @BindView(R.id.detail_walk_summary_vertline)
    View line;

    @BindDrawable(R.drawable.ic_expand_less_black_24dp)
    Drawable less;
    @BindDrawable(R.drawable.ic_expand_more_black_24dp)
    Drawable more;

    @OnClick(R.id.detail_walk_summary)
    void onClick() {
        toggleExpand();
    }

    WalkSummary(Connection con,
                JourneyUtil.Section section,
                Walk w,
                ViewGroup parent,
                LayoutInflater inflater, boolean expanded) {
        this.section = section;
        this.expanded = expanded;
        activity = (DetailClickHandler) inflater.getContext();
        layout = inflater.inflate(R.layout.detail_walk_summary, parent, false);
        ButterKnife.bind(this, layout);

        JourneyUtil.setBackgroundColor(inflater.getContext(), line, JourneyUtil.WALK_CLASS);

        long dep = con.stops(section.from).departure().scheduleTime();
        long arr = con.stops(section.to).arrival().scheduleTime();
        long durationMinutes = (arr - dep) / 60;
        String durationString = durationMinutes == 0 ? "" : String.format(summaryDurationTemplate,
                TimeUtil.formatDuration(
                        durationMinutes));
        summary.setText(String.format(summaryTemplate, durationString));

        setupIcon();
    }

    void toggleExpand() {
        if (expanded) {
            activity.contractSection(section);
        } else {
            activity.expandSection(section);
        }
    }

    void setupIcon() {
        upper.setVisibility(View.VISIBLE);
        lower.setVisibility(View.VISIBLE);
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
