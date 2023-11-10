# group: nodal_properties
#   group-components: 480

rule nodal_properties:
    input:
        graph=inputs["connectome"].path
    output:
        bids(
            source / dataset,
            suffix="graphprops.tsv",
            **inputs["connectome"].wildcards,
        )
    log: f"logs/nodal_properties/{'.'.join(inputs['connectome'].wildcards.values())}.log"
    benchmark: f"benchmarks/nodal_properties/{'.'.join(inputs['connectome'].wildcards.values())}.tsv"
    group: "nodal_properties"
    threads: 2
    resources:
        mem_mb=6000,
        runtime=6,
    envmodules:
        "python/3.10"
    shell:
        boost(
            nodal_props_venv.script,
            pyscript(
                script="scripts/nodal_properties.py",
                input=['graph'],
                wildcards=list(inputs["connectome"].wildcards),
            )
        )

rule merge_nodal_properties:
    input:
        inputs['connectome'].expand(
            rules.nodal_properties.output,
            allow_missing=True,
        )
    output:
        Path(config['output_dir'], f'{dataset}_nodes.tsv',)
    threads: 1
    run:
        pd.concat(
            pd.read_csv(i, sep="\t") for i in input
        ).reset_index(drop=True).to_csv(str(output), sep="\t", index=False)

