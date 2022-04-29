import xml.etree.ElementTree as ET
import yaml
import nmf_gym
from pathlib import Path


if __name__ == '__main__':
    df3dpp_to_nmf_names = {
        'ThC_pitch': 'Coxa', 'ThC_roll': 'Coxa_roll', 'ThC_yaw': 'Coxa_yaw',
        'CTr_pitch': 'Femur', 'CTr_roll': 'Femur_roll',
        'FTi_pitch': 'Tibia', 'TiTa_pitch': 'Tarsus1'
    }
    nmf_to_df3dpp_names = {v: k for k, v in df3dpp_to_nmf_names.items()}
    
    # The default position of certain fixed joints as specified in the
    # mesh files lead to unnatural poses. They should be set to non-zero
    # fixed positions for better visualization. Counterintuitively, to
    # be set to such fixed positions, they need to be unfixed in the
    # SDF. Here's a set of such joints.
    fixed_joints_to_unfix = {
        'joint_A3', 'joint_A4', 'joint_A5', 'joint_A6',
        'joint_LAntenna', 'joint_RAntenna',
        'joint_Rostrum', 'joint_Haustellum', 'joint_Head',
        'joint_LWing_roll', 'joint_LWing_yaw',
        'joint_RWing_roll', 'joint_RWing_yaw'
    }
    
    # Load SDF
    basedir = Path(nmf_gym.__path__[0]).absolute().parent
    sdf = ET.parse(basedir / 'data/design/sdf/neuromechfly_42dof.sdf')
    root = sdf.getroot()

    # Load joint limits
    with open(basedir / 'data/pose/joint_limits.yaml') as f:
        joint_limits_dict = yaml.safe_load(f)

    # Modify joint limits in SDF
    model = root[0]
    for child in model:
        if child.tag == 'joint' and child.attrib['type'] == 'revolute':
            joint_name = child.attrib['name'].replace('joint_', '')
            side = joint_name[0]
            pos = joint_name[1]
            nmf_dof_name = joint_name[2:]
            df3dpp_dof_name = nmf_to_df3dpp_names[nmf_dof_name]
            vmin = joint_limits_dict[pos][df3dpp_dof_name]['min']
            vmax = joint_limits_dict[pos][df3dpp_dof_name]['max']
            if side == 'R':
                if (nmf_dof_name.endswith('_roll') or
                        nmf_dof_name.endswith('_yaw')):
                    vmin *= -1
                    vmax *= -1
            if vmax < vmin:
                vmin, vmax = vmax, vmin
            print(f'{joint_name}: limit set to ({vmin}, {vmax})')
            limit = child.find('axis').find('limit')
            limit.find('lower').text = str(vmin)
            limit.find('upper').text = str(vmax)

    # Save new SDF
    sdf.write(
        basedir / 'data/design/sdf/neuromechfly_42dof_with_limit.sdf',
        xml_declaration=True
    )

    # Also save a SDF where joints (wings etc) whose angles are fixed to
    # specified angles are unfixed (ie revolute type) in the SDF. This is
    # for better visualization only.
    for child in model:
        if (child.tag == 'joint' and
                child.attrib['name'] in fixed_joints_to_unfix):
            child.attrib['type'] = 'revolute'
    sdf.write(
        basedir / 'data/design/sdf/neuromechfly_42dof_with_limit_viz.sdf',
        xml_declaration=True
    )