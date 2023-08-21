package de.motis_project.app2.journey;

import android.view.LayoutInflater;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.TextView;

import androidx.annotation.Nullable;

import butterknife.BindString;
import butterknife.BindView;
import butterknife.OnClick;
import de.motis_project.app2.R;
import de.motis_project.app2.TimeUtil;
import de.motis_project.app2.io.Status;
import de.motis_project.app2.io.error.MotisErrorException;
import motis.lookup.LookupScheduleInfoResponse;
import rx.Observable;
import rx.android.schedulers.AndroidSchedulers;
import rx.schedulers.Schedulers;

public class JourneyErrorViewHolder extends JourneyViewHolder {

    @BindView(R.id.journey_loading_error_message)
    TextView messageView;

    @BindView(R.id.journey_loading_error_retry)
    Button retryButton;

    @OnClick(R.id.journey_loading_error_retry)
    void onClick() {
    }

    @BindString(R.string.schedule_range)
    String scheduleRangeTemplate;

    @BindString(R.string.routing_error)
    String routingErrorMessage;

    private final Observable<LookupScheduleInfoResponse> scheduleInfo = Status.get().getServer().scheduleInfo();

    public JourneyErrorViewHolder(ViewGroup parent, LayoutInflater inflater, MotisErrorException mee) {
        super(inflater.inflate(R.layout.journey_loading_error, parent, false), inflater);
        messageView.setText(buildMessage(mee, scheduleInfo.toBlocking().firstOrDefault(null)));
        scheduleInfo.subscribeOn(Schedulers.io())
                .observeOn(AndroidSchedulers.mainThread())
                .subscribe(lookupScheduleInfoResponse ->
                                messageView.setText(JourneyErrorViewHolder.buildScheduleRangeError(
                                        lookupScheduleInfoResponse, scheduleRangeTemplate, routingErrorMessage)),
                        Throwable::printStackTrace);
    }

    public JourneyErrorViewHolder(ViewGroup parent, LayoutInflater inflater, int msgId) {
        super(inflater.inflate(R.layout.journey_loading_error, parent, false), inflater);
        messageView.setText(inflater.getContext().getText(msgId));
    }

    private static String messageFromMotisError(MotisErrorException mee) {
        return mee.category + ": " + mee.reason;
    }

    private String buildMessage(MotisErrorException mee, LookupScheduleInfoResponse scheduleInfo) {
        return buildMessage(mee, scheduleInfo, scheduleRangeTemplate, routingErrorMessage);
    }

    public static String buildMessage(MotisErrorException mee,
                                      @Nullable LookupScheduleInfoResponse scheduleInfo,
                                      String scheduleRangeTemplate, String routingErrorMessage) {
        if (mee.category.equals("motis::routing") && mee.code == 4) {
            return buildScheduleRangeError(scheduleInfo, scheduleRangeTemplate, routingErrorMessage);
        } else {
            return messageFromMotisError(mee);
        }
    }

    public static String buildScheduleRangeError(@Nullable LookupScheduleInfoResponse scheduleInfo,
                                                 String scheduleRangeTemplate, String routingErrorMessage) {
        String scheduleInfoStr = scheduleInfo != null ? String.format(scheduleRangeTemplate,
                TimeUtil.formatDate(scheduleInfo.begin()),
                TimeUtil.formatDate(scheduleInfo.end())) : "";
        return routingErrorMessage + "\n" + scheduleInfoStr;
    }
}
