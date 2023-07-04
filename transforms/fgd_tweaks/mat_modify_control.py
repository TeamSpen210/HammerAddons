"""Tweak material_modify_control to avoid the variable showing up in instances."""
import srctools.logger
from hammeraddons.bsp_transform import trans, Context
from srctools.tokenizer import TokenSyntaxError
from srctools.vmt import Material


LOGGER = srctools.logger.get_logger(__name__)


@trans('FGD - material_modify_control')
def material_modify_control(ctx: Context) -> None:
    """Prepend $ to mat-mod-control variable keyvalues if required.

    This allows Hammer to not detect this as a fixup variable.
    """
    # Material name -> does it have either materialmodify or materialmodifyanimated proxies.
    mat_has_proxy: dict[str, bool] = {}
    # Model or "*xx" index -> list of relevant materials.
    model_mats: dict[str, list[str]] = {}
    filter_fsys = ctx.pack.fsys  # Close over just filesystem.

    def filter_materials(mat_name: str) -> bool:
        """Check if this material has the proxy."""
        lowered = mat_name.casefold()
        if not lowered.startswith(('materials/', 'materials\\')):
            mat_name = f'materials/{mat_name}'
            lowered = f'materials/{lowered}'
        if not lowered.endswith('.vmt'):
            mat_name += '.vmt'
            lowered += '.vmt'
        try:
            return mat_has_proxy[lowered]
        except KeyError:
            pass
        try:
            file = filter_fsys[mat_name]
        except FileNotFoundError:
            LOGGER.warning('Material "{}" does not exist!', mat_name)
            mat_has_proxy[lowered] = False
            return False
        with file.open_str() as f:
            try:
                mat = Material.parse(f, mat_name)
                mat.apply_patches(filter_fsys)
            except (TokenSyntaxError, ValueError) as exc:
                LOGGER.warning('Material "{}" is not valid: ', exc_info=exc)
        for proxy in mat.proxies:
            if proxy.name in ('materialmodify', 'materialmodifyanimated'):
                mat_has_proxy[lowered] = True
                return True
        mat_has_proxy[lowered] = False
        return False

    for ent in ctx.vmf.by_class['material_modify_control']:
        var_name = ent['materialvar']
        if var_name and not var_name.startswith(('$', '%')):
            ent['materialvar'] = '$' + var_name
