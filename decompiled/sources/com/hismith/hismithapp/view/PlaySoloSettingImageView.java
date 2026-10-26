package com.hismith.hismithapp.view;

import android.content.Context;
import android.util.AttributeSet;
import android.view.View;
import androidx.appcompat.widget.AppCompatImageView;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.dialog.PlayModeSettingDialog;
import com.hismith.hismithapp.utils.AppManager;

/* loaded from: classes3.dex */
public class PlaySoloSettingImageView extends AppCompatImageView {
    private PlayModeSettingDialog playModeSettingDialog;

    public PlaySoloSettingImageView(Context context) {
        this(context, null);
    }

    public PlaySoloSettingImageView(Context context, AttributeSet attributeSet) {
        this(context, attributeSet, 0);
    }

    public PlaySoloSettingImageView(Context context, AttributeSet attributeSet, int i) {
        super(context, attributeSet, i);
        init();
    }

    private void init() {
        setImageResource(R.mipmap.ic_iplay_setting);
        setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.view.PlaySoloSettingImageView.1
            @Override // android.view.View.OnClickListener
            public void onClick(View view) {
                PlaySoloSettingImageView.this.showPlaySettingDialog();
            }
        });
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void showPlaySettingDialog() {
        PlayModeSettingDialog playModeSettingDialog = this.playModeSettingDialog;
        if (playModeSettingDialog == null) {
            playModeSettingDialog = new PlayModeSettingDialog(AppManager.getInstance().currentActivity(), new PlayModeSettingDialog.onStatusListener() { // from class: com.hismith.hismithapp.view.PlaySoloSettingImageView.2
                @Override // com.hismith.hismithapp.dialog.PlayModeSettingDialog.onStatusListener
                public void onDismiss() {
                    PlaySoloSettingImageView.this.setImageResource(R.mipmap.ic_iplay_setting);
                }

                @Override // com.hismith.hismithapp.dialog.PlayModeSettingDialog.onStatusListener
                public void onShow() {
                    PlaySoloSettingImageView.this.setImageResource(R.mipmap.ic_iplay_setting_select);
                }
            }, true, false);
            this.playModeSettingDialog = playModeSettingDialog;
        }
        playModeSettingDialog.show();
    }
}
