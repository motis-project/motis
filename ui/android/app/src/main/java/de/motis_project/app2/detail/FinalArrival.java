package de.motis_project.app2.detail;

import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import butterknife.BindColor;
import butterknife.BindString;
import butterknife.BindView;
import butterknife.ButterKnife;
import butterknife.OnClick;
import de.motis_project.app2.JourneyUtil;
import de.motis_project.app2.R;
import de.motis_project.app2.Str;
import de.motis_project.app2.TimeUtil;
import de.motis_project.app2.journey.ConnectionWrapper;
import motis.Connection;
import motis.EventInfo;
import motis.Stop;
import motis.TimestampReason;
import motis.Transport;

public class FinalArrival implements DetailViewHolder {
    private View layout;

    @BindString(R.string.track) String track;

    @BindView(R.id.detail_final_arrival_time) TextView arrivalTime;
    @BindView(R.id.detail_final_arrival_station) TextView arrivalStation;
    @BindView(R.id.detail_final_arrival_track) TextView arrivalTrack;
    @BindView(R.id.detail_transport_final_arrival_vertline) View line;
    @BindView(R.id.detail_transport_final_arrival_bullet) View bullet;
    @BindView(R.id.detail_final_arrival_delay_placeholder) View delayPlaceholder;
    @BindView(R.id.detail_final_arrival_delay) TextView delay;

    @BindColor(R.color.delayed) int colorRed;
    @BindColor(R.color.ontime) int colorGreen;

    protected final Stop stop;
    protected final DetailClickHandler clickHandler;

    FinalArrival(ConnectionWrapper conWrapper,
                 JourneyUtil.Section section,
                 ViewGroup parent,
                 LayoutInflater inflater) {
        clickHandler = (DetailClickHandler) inflater.getContext();
        layout = inflater.inflate(R.layout.detail_final_arrival, parent, false);
        ButterKnife.bind(this, layout);

        Connection con = conWrapper.getConnection();
        Context context = inflater.getContext();
        Transport transport = JourneyUtil.getTransport(con, section);
        long clasz = (transport != null) ? transport.clasz() : JourneyUtil.WALK_CLASS;
        JourneyUtil.setBackgroundColor(context, line, clasz);
        JourneyUtil.tintBackground(context, bullet, clasz);

        stop = con.stops(section.to);
        EventInfo arr = stop.arrival();
        arrivalTime.setText(TimeUtil.formatTime(arr.scheduleTime()));
        String stopName = stop.station().name();
        if (stopName.equals("START")) {
            stopName = conWrapper.getStartName();
        } else if (stopName.equals("END")) {
            stopName = conWrapper.getDestinationName();
        }
        arrivalStation.setText(stopName);

        String arrTrackStr = Str.san(arr.track());
        if (arrTrackStr.isEmpty()) {
            arrivalTrack.setVisibility(View.GONE);
        } else {
            arrivalTrack.setText(String.format(track, arrTrackStr));
        }

        if (arr.reason() == TimestampReason.SCHEDULE) {
            delayPlaceholder.setVisibility(View.VISIBLE);
        } else {
            delay.setText(TimeUtil.delayString(arr));
            delay.setTextColor(TimeUtil.delay(arr) ? colorRed : colorGreen);
            delay.setVisibility(View.VISIBLE);
        }
    }

    @Override
    public View getView() {
        return layout;
    }

    @OnClick(R.id.detail_final_arrival)
    public void onClick() {
        clickHandler.transportStopClicked(stop);
    }
}
