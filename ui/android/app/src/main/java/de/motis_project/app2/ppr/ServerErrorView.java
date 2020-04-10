package de.motis_project.app2.ppr;

import android.content.Context;
import android.util.AttributeSet;
import android.widget.TextView;

import butterknife.BindString;
import butterknife.ButterKnife;
import de.motis_project.app2.R;
import de.motis_project.app2.io.error.MotisErrorException;


public class ServerErrorView extends TextView {
    @BindString(R.string.ppr_empty_response)
    String emptyResponseMessage;

    public ServerErrorView(Context context) {
        super(context);
        ButterKnife.bind(this);
    }

    public ServerErrorView(Context context, AttributeSet attrs) {
        super(context, attrs);
        ButterKnife.bind(this);
    }

    public void setErrorCode(MotisErrorException mee) {
        setText(buildMessage(mee));
    }

    public void setEmptyResponse() {
        setText(emptyResponseMessage);
    }


    private static String messageFromMotisError(MotisErrorException mee) {
        return mee.category + ": " + mee.reason;
    }

    public static String buildMessage(MotisErrorException mee) {
        return messageFromMotisError(mee);
    }
}
