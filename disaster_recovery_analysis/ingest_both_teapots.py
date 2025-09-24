if __name__ == "__main__":
    import logging
    import os

    from segy.schema import Endianness
    from segy.schema import HeaderField
    from segy.standards import get_segy_standard

    import mdio
    from mdio.builder.template_registry import TemplateRegistry

    logging.getLogger("segy").setLevel(logging.DEBUG)

    os.environ["MDIO__IMPORT__CLOUD_NATIVE"] = "true"
    os.environ["MDIO__IMPORT__CPU_COUNT"] = "16"
    os.environ["MDIO__DO_RAW_HEADERS"] = "1"

    custom_headers = [
        HeaderField(name="inline", byte=181, format="int32"),
        HeaderField(name="crossline", byte=185, format="int32"),
        HeaderField(name="cdp_x", byte=81, format="int32"),
        HeaderField(name="cdp_y", byte=85, format="int32"),
    ]

    big_endian_spec = get_segy_standard(0)
    big_endian_spec.endianness = Endianness.BIG
    little_endian_spec = get_segy_standard(0)
    little_endian_spec.endianness = Endianness.LITTLE
    big_endian_spec = big_endian_spec.customize(trace_header_fields=custom_headers)
    little_endian_spec = little_endian_spec.customize(trace_header_fields=custom_headers)

    mdio.segy_to_mdio(
        segy_spec=big_endian_spec,
        mdio_template=TemplateRegistry().get("PostStack3DTime"),
        input_path="filt_mig_IEEE_BigEndian_Rev1.sgy",
        output_path="filt_mig_IEEE_BigEndian_Rev1.mdio",
        overwrite=True,
    )
    mdio.segy_to_mdio(
        segy_spec=little_endian_spec,
        mdio_template=TemplateRegistry().get("PostStack3DTime"),
        input_path="filt_mig_IEEE_LittleEndian_Rev1.sgy",
        output_path="filt_mig_IEEE_LittleEndian_Rev1.mdio",
        overwrite=True,
    )
