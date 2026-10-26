package com.hismith.hismithapp.view;

import android.content.Context;
import android.util.AttributeSet;
import android.view.View;
import androidx.appcompat.widget.AppCompatImageView;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.dialog.PlayModeSettingDialog;
import com.hismith.hismithapp.product.ProductUtil;
import com.hismith.hismithapp.utils.AppManager;

/* loaded from: classes3.dex */
public class PlaySettingImageView extends AppCompatImageView {
    private int mSMSType;
    private PlayModeSettingDialog playModeSettingDialog;

    public PlaySettingImageView(Context context) {
        this(context, null);
    }

    public PlaySettingImageView(Context context, AttributeSet attributeSet) {
        this(context, attributeSet, 0);
    }

    public PlaySettingImageView(Context context, AttributeSet attributeSet, int i) {
        super(context, attributeSet, i);
        this.mSMSType = 1;
        init();
    }

    private void init() {
        if (ProductUtil.isSMSType() && this.mSMSType == 1) {
            setVisibility(0);
        } else {
            setVisibility(4);
        }
        setImageResource(R.mipmap.btn_mode_edit_setting_white);
        setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.view.PlaySettingImageView.1
            @Override // android.view.View.OnClickListener
            public void onClick(View view) {
                PlaySettingImageView.this.showPlaySettingDialog();
            }
        });
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void showPlaySettingDialog() {
        PlayModeSettingDialog playModeSettingDialog = this.playModeSettingDialog;
        if (playModeSettingDialog == null) {
            playModeSettingDialog = new PlayModeSettingDialog(AppManager.getInstance().currentActivity(), new PlayModeSettingDialog.onStatusListener() { // from class: com.hismith.hismithapp.view.PlaySettingImageView.2
                @Override // com.hismith.hismithapp.dialog.PlayModeSettingDialog.onStatusListener
                public void onDismiss() {
                    PlaySettingImageView.this.setImageResource(R.mipmap.btn_mode_edit_setting_white);
                }

                @Override // com.hismith.hismithapp.dialog.PlayModeSettingDialog.onStatusListener
                public void onShow() {
                    PlaySettingImageView.this.setImageResource(R.mipmap.btn_mode_edit_setting_purple);
                }
            }, false, true);
            this.playModeSettingDialog = playModeSettingDialog;
        }
        playModeSettingDialog.show();
    }

    public void setmSMSType(int i) {
        this.mSMSType = i;
        init();
    }
}
