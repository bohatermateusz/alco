package com.hismith.hismithapp;

import android.os.Bundle;
import android.widget.TextView;
import com.hismith.hismithapp.http.RetrofitRequest;
import com.hismith.hismithapp.http.RxSubscribe;
import com.hismith.hismithapp.model.PlayLogsListModel;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.SharedPreferencesUtil;
import io.reactivex.android.schedulers.AndroidSchedulers;
import io.reactivex.schedulers.Schedulers;

/* loaded from: classes3.dex */
public class PlayStatisticsActivity extends BaseActivity {
    private RetrofitRequest baseRequest;
    private String requestId = "";
    private TextView title_txt_name;

    private void init() {
        this.baseRequest = new RetrofitRequest();
        playLogs(1);
    }

    private void initView() {
        TextView textView = (TextView) findViewById(R.id.title_txt_name);
        this.title_txt_name = textView;
        LanguageUtil.setText(this, textView, "playRecord");
    }

    private void setListener() {
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.fragment.app.FragmentActivity, androidx.activity.ComponentActivity, androidx.core.app.ComponentActivity, android.app.Activity
    protected void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        setContentView(R.layout.activity_play_statistics);
        initView();
        init();
        setListener();
    }

    public void playLogs(int i) {
        this.baseRequest.getApi().playLogs((String) SharedPreferencesUtil.get(this, Constans.Token), i, this.requestId).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<PlayLogsListModel>(this, null, false) { // from class: com.hismith.hismithapp.PlayStatisticsActivity.1
            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(PlayLogsListModel playLogsListModel) {
            }
        });
    }
}
