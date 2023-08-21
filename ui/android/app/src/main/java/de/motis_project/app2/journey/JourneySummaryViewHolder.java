package de.motis_project.app2.journey;

import android.content.Context;
import android.graphics.drawable.Drawable;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.LinearLayout;
import android.widget.TextView;

import androidx.core.graphics.drawable.DrawableCompat;

import butterknife.BindColor;
import butterknife.BindView;
import de.motis_project.app2.JourneyUtil;
import de.motis_project.app2.R;
import de.motis_project.app2.TimeUtil;
import motis.EventInfo;
import motis.TimestampReason;

public class JourneySummaryViewHolder extends JourneyViewHolder {
    enum ViewMode {LONG, SHORT, OFF}

    @BindView(R.id.dep_sched_time)
    TextView depSchedTime;

    @BindView(R.id.dep_time)
    TextView depTime;

    @BindView(R.id.arr_sched_time)
    TextView arrSchedTime;

    @BindView(R.id.arr_time)
    TextView arrTime;

    @BindView(R.id.duration)
    TextView duration;

    @BindView(R.id.transports)
    LinearLayout transports;

    @BindView(R.id.dot)
    View dotView;


    @BindColor(R.color.delayed)
    int colorRed;
    @BindColor(R.color.ontime)
    int colorGreen;

    Context context;

    public JourneySummaryViewHolder(ViewGroup parent, LayoutInflater inflater) {
        super(inflater.inflate(R.layout.journey_list_item, parent, false), inflater);
        this.context = parent.getContext();
    }

    void setConnection(ConnectionWrapper con) {
        EventInfo dep = con.getFirstStop().departure();
        EventInfo arr = con.getLastStop().arrival();

        long minutes = (arr.scheduleTime() - dep.scheduleTime()) / 60;
        duration.setText(TimeUtil.formatDuration(minutes));

        if (con.isIntermodal()) {
            Drawable bg = DrawableCompat.wrap(dotView.getBackground());
            DrawableCompat.setTint(bg.mutate(), con.getColor(context));
            dotView.setBackground(bg);
            dotView.setVisibility(View.VISIBLE);
        } else {
            dotView.setVisibility(View.GONE);
        }

        depSchedTime.setText(TimeUtil.formatTime(dep.scheduleTime()));
        depTime.setText(TimeUtil.formatTime(dep.time()));

        arrSchedTime.setText(TimeUtil.formatTime(arr.scheduleTime()));
        arrTime.setText(TimeUtil.formatTime(arr.time()));

        if (dep.reason() != TimestampReason.SCHEDULE) {
            depTime.setTextColor(TimeUtil.delay(dep) ? colorRed : colorGreen);
        }
        if (arr.reason() != TimestampReason.SCHEDULE) {
            arrTime.setTextColor(TimeUtil.delay(arr) ? colorRed : colorGreen);
        }

        TransportViewCreator.addTransportViews(
                JourneyUtil.getTransports(con.getConnection(), true),
                inflater, transports);
    }
}
